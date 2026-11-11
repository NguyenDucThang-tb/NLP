"""kiểm thử kỹ thuật; dữ liệu fixture ở đây không phải kết quả thí nghiệm IMDb."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, DistilBertConfig
from pretrained_classification import (
    Config, LabRunner, build_model, compute_metrics, count_parameters, create_stratified_splits,
    create_training_subset, extract_error_examples, freeze_encoder, load_tokenizer,
    plot_confidence_analysis, save_experiment_result, validate_text_frame,
)


LOCAL_ONLY = os.environ.get("HF_HUB_OFFLINE") == "1"


class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def tiny_model(self):
        return AutoModelForSequenceClassification.from_config(DistilBertConfig(
            vocab_size=100, dim=16, hidden_dim=32, n_layers=1, n_heads=2,
            num_labels=2, dropout=0, attention_dropout=0, seq_classif_dropout=0))

    def test_config_rejects_partial_full(self):
        with self.assertRaises(ValueError): Config(mode="full", train_size=20)
        self.assertEqual(Config(mode="quick").train_size, 2000)

    def test_nested_and_reproducible_subsets(self):
        frame = pd.DataFrame({"text":[f"fixture {i}" for i in range(80)],
                              "label":[i%2 for i in range(80)],"source_id":[str(i) for i in range(80)]})
        previous=set()
        for f in [.1,.25,.5,1]:
            subset=create_training_subset(frame,f)
            self.assertTrue(previous.issubset(set(subset.source_id)))
            self.assertEqual(set(subset.label),{0,1})
            pd.testing.assert_frame_equal(subset,create_training_subset(frame,f))
            previous=set(subset.source_id)

    def test_splits_disjoint_and_official_test_preserved(self):
        def fixture(prefix,n):
            return pd.DataFrame({"text":[f"{prefix} {i}" for i in range(n)],
                                 "label":[i%2 for i in range(n)],"source_id":[f"{prefix}-{i}" for i in range(n)]})
        raw={"train":fixture("train",100),"test":fixture("test",40)}
        s=create_stratified_splits(raw,Config(mode="full"))
        self.assertEqual([len(s[k]) for k in ["train","validation","test"]],[80,20,40])
        ids=[set(f.source_id) for f in s.values()]
        self.assertFalse(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
        self.assertEqual(set(s["test"].source_id),set(raw["test"].source_id))

    def test_missing_or_invalid_labels_stop(self):
        for labels,text in [([0,None],["a","b"]),([0,3],["a","b"]),([0,1],["a"," "])]:
            with self.assertRaises(ValueError):
                validate_text_frame(pd.DataFrame({"text":text,"label":labels}),"fixture")
        runner=object.__new__(LabRunner)
        runner.config=Config(mode="smoke")
        runner.run_model=lambda *args: {}
        with patch("pretrained_classification.load_dataset",side_effect=RuntimeError("network unavailable")):
            with self.assertRaises(OSError): runner.domain_shift()

    def test_real_tokenizer_shapes_padding_ids_mapping(self):
        tokenizer=load_tokenizer(local_files_only=LOCAL_ONLY)
        batch=tokenizer(["The cats are playing.","Bad."],padding=True,return_tensors="pt")
        self.assertEqual(batch.input_ids.shape,batch.attention_mask.shape)
        self.assertTrue(torch.equal(batch.input_ids==tokenizer.pad_token_id,batch.attention_mask==0))
        self.assertTrue(((batch.input_ids>=0)&(batch.input_ids<len(tokenizer))).all())
        self.assertIsNotNone(tokenizer("The cats are playing.").word_ids())

    def test_forward_probabilities_and_contextual_representation(self):
        model=self.tiny_model().eval()
        with torch.inference_mode():
            out=model(input_ids=torch.tensor([[1,2,3],[1,4,0]]),
                      attention_mask=torch.tensor([[1,1,1],[1,1,0]]), output_hidden_states=True)
        self.assertEqual(tuple(out.logits.shape),(2,2))
        p=torch.softmax(out.logits,1)
        self.assertTrue(((p>=0)&(p<=1)).all())
        self.assertTrue(torch.allclose(p.sum(1),torch.ones(2)))
        self.assertEqual(tuple(out.hidden_states[-1].shape),(2,3,16))

    def test_masked_padding_does_not_change_classification(self):
        model=self.tiny_model().eval()
        with torch.inference_mode():
            short=model(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.tensor([[1,1,1]])).logits
            padded=model(input_ids=torch.tensor([[1,2,3,0,0]]),attention_mask=torch.tensor([[1,1,1,0,0]])).logits
        self.assertTrue(torch.allclose(short,padded,atol=1e-6))

    def test_frozen_and_full_start_from_same_real_checkpoint_and_head(self):
        cfg=Config(mode="smoke",local_files_only=LOCAL_ONLY)
        full=build_model(cfg,"pretrained","fine_tuning")
        self.assertTrue(count_parameters(full)["encoder_trainable"])
        reference={n:p.detach().clone() for n,p in full.named_parameters()
                   if n.startswith(("classifier.","pre_classifier.")) or n.endswith("word_embeddings.weight")}
        del full
        frozen=build_model(cfg,"pretrained","frozen")
        self.assertFalse(count_parameters(frozen)["encoder_trainable"])
        for n,p in frozen.named_parameters():
            if n in reference: self.assertTrue(torch.equal(reference[n],p),n)
        del frozen

    def test_frozen_encoder_unchanged_but_head_updates(self):
        model=freeze_encoder(self.tiny_model())
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        model(input_ids=torch.tensor([[1,2,3],[1,4,5]]),labels=torch.tensor([0,1])).loss.backward()
        optimizer.step()
        for n,p in model.base_model.named_parameters():
            self.assertFalse(p.requires_grad)
        encoder_ids={id(p) for p in model.base_model.parameters()}
        self.assertTrue(all(torch.equal(before[n],p) for n,p in model.named_parameters() if id(p) in encoder_ids))
        self.assertTrue(any(not torch.equal(before[n],p) for n,p in model.named_parameters() if id(p) not in encoder_ids))

    def test_scratch_never_calls_weight_loader(self):
        cfg=Config(mode="smoke",local_files_only=LOCAL_ONLY)
        with patch("pretrained_classification.AutoModelForSequenceClassification.from_pretrained",side_effect=AssertionError("weight loader called")):
            model=build_model(cfg,"scratch")
        self.assertEqual(model.config.dim,768)
        self.assertEqual(model.config.n_layers,6)
        self.assertEqual(count_parameters(model)["total_parameters"],66955010)
        del model

    def test_metrics_and_missing_class(self):
        metrics=compute_metrics([0,0,1,1],[0,0,0,1])
        self.assertAlmostEqual(metrics["accuracy"],.75)
        self.assertAlmostEqual(metrics["f1_macro"],(.8+2/3)/2)
        self.assertTrue(all(np.isfinite(v) for v in compute_metrics([0,0],[0,0]).values()))

    def test_result_upsert_and_reject_invalid_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"results.csv"
            row={"experiment_id":"14","run_id":"fixture","split":"validation","domain":"fixture",
                 "accuracy":.5,"precision_macro":.5,"recall_macro":.5,"f1_macro":.5}
            save_experiment_result(path,row); save_experiment_result(path,row)
            self.assertEqual(len(pd.read_csv(path)),1)
            with self.assertRaises(ValueError): save_experiment_result(path,{**row,"accuracy":float("nan")})
            self.assertEqual(len(pd.read_csv(path)),1)

    def test_confidence_empty_groups_and_error_shortages(self):
        labels=np.array([0,0,1,1]); pred=np.array([1,0,0,1])
        proba=np.array([[.2,.8],[.6,.4],[.7,.3],[.1,.9]])
        cfg=Config(mode="smoke")
        frame=pd.DataFrame({"text":["fixture"]*4,"source_id":list("abcd")})
        _,selected,availability=extract_error_examples(frame,labels,pred,proba,cfg)
        self.assertEqual(len(selected),2)
        self.assertEqual(availability["FP"]["missing"],2)
        self.assertEqual(availability["high_confidence_wrong"]["missing"],2)
        with tempfile.TemporaryDirectory() as tmp:
            groups=plot_confidence_analysis(labels,pred,proba,Path(tmp)/"plot.png",cfg)
            self.assertEqual(int(groups["count"].sum()),4)
            self.assertEqual(int(groups["incorrect"].sum()),2)
            empty=plot_confidence_analysis(labels,pred,np.array([[.6,.4]]*4),Path(tmp)/"empty.png",cfg)
            high=empty[empty.group=="High"].iloc[0]
            self.assertEqual(high["count"],0)
            self.assertTrue(pd.isna(high["accuracy"]))

    def test_notebooks_valid_and_code_compiles(self):
        paths=list(Path(__file__).parent.glob("*.ipynb"))
        self.assertEqual(len(paths),2)
        for path in paths:
            notebook=nbformat.read(path,as_version=4); nbformat.validate(notebook)
            for i,c in enumerate(notebook.cells):
                if c.cell_type=="code": compile(c.source,f"{path.name}:{i}","exec")


if __name__ == "__main__":
    unittest.main(verbosity=2)
