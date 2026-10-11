<!-- CODEX_ERROR_ANALYSIS_START -->
# LAB 04 — Error Analysis

Selected model: TF-IDF + Logistic Regression
Validation metrics: Accuracy 0.8868; Precision 0.8801; Recall 0.8956; F1 0.8878.
Correct: 8868; incorrect: 1132; FP: 610; FN: 522.
Negative error rate: 0.1220; positive error rate: 0.1044.
Most frequent error direction: FP.
Training class distribution (Negative=0, Positive=1): {0: 15000, 1: 15000}.

Possible causes are observable-clue hypotheses, not proven causes. Student verification is pending.

## Example 8479 — Correct
True label: 1; predicted label: 1; score: p(positive)=0.6924.
Evidence: Word count: 335; Negation: not; Positive cues: great; Negative cues: None detected; Low-frequency/OOV: quatermass, carlson, beleaguered, brits, spoils, realising, untill, hid, bedrooms, pyrotechnics
Neutral technical description: Predicted 1 while actual label is 1; p(positive)=0.6924.
Possible causes (hypotheses): negation present; low-frequency or OOV terms
Student-verified Cause: Pending student verification.

Complete original review:

LIFEFORCE is an extremely schizophrenic movie, based on Colin Wilson`s novel The Space Vampires the script ignores most of the novel`s concepts and structure ( Indeed it owes more to the QUATERMASS serials than the novel ) but the scenes it does leave in from the novel are nearly identical to those in the film . And talking of the script it must be one of the most uneven in cinema history , it`s though it was written in chapters by several different people. Take for instance Carlson , he disappears after the early shuttle scenes which led me to believe he was dead then he turns up again halfway through the film in order to explain the plot to the beleaguered Brits and it`s this lack of attention by the screenwriters that spoils the film . And there`s plenty of other clumsy scripting such as the heroes returning to London in a helicopter and not realising it has been over run by zombies untill they`re flying over it .<br /><br />I could go on at great length about these plot holes but LIFEFORCE is actually enjoyable to watch as long as you don`t use your brain . It`s good to see a sci-fi horror film from an era when aliens were portrayed as being cute creatures that children hid in their bedrooms so that nasty human adults wouldn`t get their hands on them . The special effects and pyrotechnics are very very good , there`s lots of action and stunts and LIFEFORCE features one of the most memorable aliens in the form of the space girl . When mentioning LIFEFORCE in conversation with males it`s always a race to say " Seen the alien in LIFEFORCE? She can suck the lifeforce out of me anytime " Hardly surprising looking at the demographics of the votes that this film is more popular with males than females<br /><br />" Don`t worry . A naked girl can`t escape from here " Can`t she ? Pity<br /><br />

## Example 1676 — Correct
True label: 0; predicted label: 0; score: p(positive)=0.2557.
Evidence: Word count: 143; Negation: not; Positive cues: None detected; Negative cues: bad; Low-frequency/OOV: cheesiest, teared, cheeks, ached, subpar, dives, whoopass, whatchugonnado, infamously, goateed
Neutral technical description: Predicted 0 while actual label is 0; p(positive)=0.2557.
Possible causes (hypotheses): negation present; low-frequency or OOV terms
Student-verified Cause: Pending student verification.

Complete original review:

... in search of the cheesiest "so bad it's good" movie, I've repeatedly laughed at the first fifteen minutes of various films, only to be left disappointed and bored at the end. Not this time!!! My eyes teared up, my belly and my cheeks ached from laughing so hard throughout the movie. Sure, Hulk Hogan is a subpar actor and the plot is utterly predictable, but everyone dives into this movie knowing all this - all anyone wants to see when renting this is Hogan breaking out a can of whoopass, with a bunch of "YEAH BROTHER"s and "WHATCHUGONNADO"s flying from his infamously goateed mouth. And while the Hulkster on the screen pales a bit in comparison to the Hulkster in the ring, seekers of the ultimate cheese will certainly not be disappointed by this backhand gem of a flick. A laugh riot.

## Example 256 — Correct
True label: 0; predicted label: 0; score: p(positive)=0.3455.
Evidence: Word count: 185; Negation: not; Positive cues: None detected; Negative cues: boring; Low-frequency/OOV: jettisoned, rustic, honed, possum, lodge, blandly, embellished, devised, plotline, offscreen
Neutral technical description: Predicted 0 while actual label is 0; p(positive)=0.3455.
Possible causes (hypotheses): negation present; low-frequency or OOV terms
Student-verified Cause: Pending student verification.

Complete original review:

Essentially this is a dreadful film with a few features that may or may not redeem it for you, depending on how much you want them to. In "opening up" "The Red Green Show" for the big screen, the filmmakers jettisoned the rustic charm carefully honed over a decade's worth of episodes set in and around Possum Lodge in favor of a blandly-photographed "road movie" with a ghastly faux-Hollywood "big" musical score and profoundly boring storyline that's not embellished with enough good gags to make it as entertaining as even the most mediocre episode of the TV series.<br /><br />Having devised a plotline that keeps most of the members of Possum Lodge offscreen for most of the film and requires virtually everyone concerned to be despicably mean to the loveable Harold (who's the hero of the film, the usually affably crusty Red being relegated in this incarnation to nothing more than the role of head Harold abuser), the only performers who really get to shine are Patrick McKenna and Peter Keleghan as Harold and Ranger Gord, who deliver satisfyingly large-screen versions of their small-screen characters.

## Example 6589 — False Positive
True label: 0; predicted label: 1; score: p(positive)=0.6612.
Evidence: Word count: 282; Negation: never, no; Positive cues: None detected; Negative cues: None detected; Low-frequency/OOV: chile, deposed, salvador, allende, marxist, democratically, chilean, capitalism, socialism, disputed
Neutral technical description: Predicted 1 while actual label is 0; p(positive)=0.6612.
Possible causes (hypotheses): negation present; low-frequency or OOV terms
Student-verified Cause: Pending student verification.

Complete original review:

28 years before 9/11, there was another 9/11 which represented a key date in the history of Chile, South America and the whole world. This was the date in 1973 when a bloody coup in Chile deposed Salvador Allende the first Marxist president elected democratically anywhere in the world and put an end to the Chilean experiment of a democratic transition from capitalism to socialism. Allende committed suicide when the armed forces attacked the presidential palace.<br /><br />Unfortunately this film is too biased and too nostalgic towards the time of Allende's rule to be an objective rendition of the man and of his place in history. The times were troubled and Allende was a disputed figure in the history of his country and of the whole world. True, he was democratically elected, but his policies plunged Chile into economic crisis. He was deposed by a coup and a right-wing dictatorship followed with repression and flagrant human rights abuses, but he was also an ally of Castro who saw in his policies another way of making revolution. We'll never know if his tentative to build a socialist yet democratic society would have succeeded. The authors of the movie take a completely pro-Allende position, there is no opinion or point of view trying to explain the other side, to answer questions like why did the middle class oppose him, or how his democratic views could go together with supporting or being supported by Castro. The tone of the commentaries is nostalgic and apologetic, almost propagandistic. People who want to get a better understanding of this episode of the history need to wait for a more balanced and objective film or book in the future.

## Example 3818 — False Negative
True label: 1; predicted label: 0; score: p(positive)=0.2743.
Evidence: Word count: 126; Negation: not; Positive cues: enjoyed; Negative cues: None detected; Low-frequency/OOV: dreading, ahh
Neutral technical description: Predicted 0 while actual label is 1; p(positive)=0.2743.
Possible causes (hypotheses): negation present; low-frequency or OOV terms
Student-verified Cause: Pending student verification.

Complete original review:

Just watched this after my mother brought it back from America for me, was dreading watching this after all the negative comments on here but I have to say, yes the acting is cheesy, some of the effects are laughable.<br /><br />But you have to remember this was meant to be 1898 not 2005, and for such a low budget I thought it was quite good. I enjoyed this version much more then the Spielberg version I saw last week.<br /><br />I have read the book so many times, and found myself going "ahh yes that's in the book" almost all the time, with the other version hardly anything of the book existed.<br /><br />So well done for at least trying to make a true version.

## Example 4816 — False Negative
True label: 1; predicted label: 0; score: p(positive)=0.0742.
Evidence: Word count: 103; Negation: not; Positive cues: None detected; Negative cues: bad, waste; Low-frequency/OOV: None detected
Neutral technical description: Predicted 0 while actual label is 1; p(positive)=0.0742.
Possible causes (hypotheses): negation present
Student-verified Cause: Pending student verification.

Complete original review:

Okay, I think we're all agreed that Michael Jackson was the low point.<br /><br />And the special effects too. But, please, keep in mind that this was NOT a big-budget film, okay? Not every film gets as much of a budget as Harry Potter or Star Wars.<br /><br />However, I thought it was pretty funny altogether. B-? Nothing that would, in my opinion, waste your time.<br /><br />Parodies are always fun to watch, and just because it wasn't big budget doesn't mean it's bad.<br /><br />I think this was a good movie, if weak at some points.<br /><br />Hope this comment helps. ~Angela
<!-- CODEX_ERROR_ANALYSIS_END -->
