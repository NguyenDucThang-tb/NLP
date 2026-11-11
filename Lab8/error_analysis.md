# Bài 20 — Error analysis

Bằng chứng lấy từ IMDb test, mode `quick`, model pretrained fine-tuned, checkpoint chọn bằng validation macro-F1.

Model: `distilbert/distilbert-base-uncased`; max_length=128; seed=42. Run ID/checkpoint chính xác nằm ở record Bài 19 trong results.csv và complete.json.

## Số lượng mẫu có thật

|                       |   available |   requested |   missing |
|:----------------------|------------:|------------:|----------:|
| FP                    |          15 |           3 |         0 |
| FN                    |          25 |           3 |         0 |
| high_confidence_wrong |           2 |           2 |         0 |

High-confidence wrong dùng ngưỡng `0.9`. Không đủ mẫu ở nhóm nào được ghi ở cột missing; không tạo mẫu bù. Toàn bộ review/predictions: `outputs/quick/test_predictions.csv`.

### imdb-test-20411 — FN

Text:

> OK firstly, if your not a fan of the whole low budget horror genre then don't bother with this. You really need to be a fan to get the gag.<br /><br />The move is basically about snakes ..on a train. Lots of them.<br /><br />There is an ancient curse involved and a crazy ending which I wont spoil. The gore effect are full on and real icky... but the movie is mainly one big gory gag about snakes scaring the hell out of a bunch of people trapped on a train.<br /><br />The suspense is built up well and there are some memorable and well composed scenes.. some of the acting is a bit hammy (watch for the surf dudes) but thats not really the point... Give this movie a go. Know what to expect and you wont regret it!

- True label: Positive
- Prediction: Negative
- Confidence: 0.92675078
- Error type: TODO(student)
- Hypothesis: TODO(student)

### imdb-test-12024 — FP

Text:

> Jack Lemmon was one of the finest actors that had ever graced the screen. He could effortlessly switch from dramatic roles to comedic with ease, making most of his peers green with envy. While his performance in "Save The Tiger" is Oscar-worthy, I feel it was given to him as he had missed out on his other opportunities to win the award due to other, better roles that had preceded this current one.<br /><br />This is also one of those pretentious movies that comes out to basically showcase the talent of the cast, or in this case, one particular member. It's too bad the screenwriter's output didn't match that of Lemmon's. Don't waste your time with this one.

- True label: Negative
- Prediction: Positive
- Confidence: 0.90631729
- Error type: TODO(student)
- Hypothesis: TODO(student)

### imdb-test-19904 — FN

Text:

> I found this one to be more chaotic than the average Stooges short (as strange as that may sound). There were several funny bits, especially the running gags ("Calling Dr. Howard, Dr. Fine, Dr. Howard!", the glass door breaking, the Stooges running into the supply room and coming back out with... well, you know), but also quite of bit of it was taken up with things that just didn't make any sense to me. I have to assume that these were generally take-offs of scenes from the film "Men In White", but since I don't know much about that movie I can't say for sure. Maybe if someone could explain these I'd appreciate this short more.

- True label: Positive
- Prediction: Negative
- Confidence: 0.89895034
- Error type: TODO(student)
- Hypothesis: TODO(student)

### imdb-test-16344 — FN

Text:

> This movie possesses something most other movies of its genre do not: intelligence and good messages. Accepted, is the story of a kid named Bartleby (Justin Long) who gets rejected by every college he applies to. His parents are incredibly strict, frustrated and disappointed. So, Bartleby and a few other students who face similar problems start a fake college, to pawn themselves off to their parents as college kids. Not everything goes to plan, and the movie is about them winging it. But behind this plot which sounds ridiculous, there are good messages, morals and a new outlook on the American educational system. The college they create "South Harmon Institute of Technology" accepts those who did not get accepted anywhere else. The message is that you do not need "money, tradition, or fancy books- you just need a desire to better yourself." they are a completely unorthodox school, but what the movie teaches is that thats still OK- and different (whether its a person or a school) is not necessarily bad, and that every one no matter should be given the chance to better themselves- despite whether they are 'weird.' its a message of non-comformity and has traces of anti-authority sprinkled in as well. But not only are messages great- its hilarious! haha i laughed very hard. the humor can be subtle but can also be blatant as well, and is a great mix. Fraternities beware.

- True label: Positive
- Prediction: Negative
- Confidence: 0.89697301
- Error type: TODO(student)
- Hypothesis: TODO(student)

### imdb-test-9115 — FP

Text:

> In the aftermath of September 11th in New York, this drama about American firefighters was conceived as a salute and tribute to their professionalism. The story is told with a series of flashbacks, where after firefighter Jack Morrison (Joaquin Phoenix) has crashed through the floor of a burning building, and only communicating with Captain Mike Kennedy (John Travolta) via the radio. The flashbacks basically show how Jack grows from being a recruit, seeing Kennedy as a father figure, to being a firehouse legend. Of course, in the present day, Jack's fellow firefighters are trying to reach him, but they are too late, and in the end, he lets them leave him, and it forwards to his funeral, where he is praised as one of the best firefighters they have known. Also starring Jacinda Barrett as Linda Morrison, Terminator 2's Robert Patrick as Lenny Richter, Morris Chestnut as Tommy Drake, Billy Burke as Dennis Gauquin, Balthazar Getty as Ray Gauquin and Tim Guinee as Tony Corrigan. The blazes of the film are ultimately all I could pay attention to and enjoy seeing the characters tackle them, the rest is a bit too chatty for my liking. Adequate!

- True label: Negative
- Prediction: Positive
- Confidence: 0.88919246
- Error type: TODO(student)
- Hypothesis: TODO(student)

### imdb-test-1328 — FP

Text:

> For those of you who have never heard of the movie until now (of which, I presume there are many lucky people who haven't), I'll summarise it for you. Ryan Gosling plays the titular character of Leland, who also serves as the film's narrator (a la Kevin Spacey in American Beauty, but without the intelligent observations on life). Leland goes to jail for stabbing a retarded kid to death, and the movie attempts to figure out why he did it. He seems to be a nice boy (if not mentally absent), and is portrayed by Gosling with a complete lack of violence, anger, or agenda (and if you're waiting for him to reveal his sinister side later in the movie, don't waste your time-- it's not that kind of movie). Once in juvenile prison, Leland goes to classes taught by Pearl Madison, ably portrayed by Don Cheadle (who is incapable of anything but quality, even when in bad movies). Pearl attempts to unlock the mystery of Leland in an attempt to figure out how such a kid could do such a thing, and so he could write a book about it later (along with being a juvenile prison teacher, Pearl is also an aspiring author).<br /><br />The relationship between Leland and Pearl is the driving narrative behind the film, as their talks unveil Leland's past to the audience. However, to call it the central focus would imply that this meandering film had one. It does not. The United States of Leland boasts an impressive cast, which seems to be to the detriment of the film. It seems as though writer/director Matthew Ryan Hodge (don't worry that you haven't heard of him, he's never done anything) had to give EVERY character a personal story arc and personality flaw in order to get the actors to play them. Most of these traits and stories are clichéd, and most go underdeveloped and unresolved.<br /><br />I'll try and break them down here: Martin Donovan and Ann Magnuson are the parents of the slain retarded boy (I love how the movie kept calling the kid "retarded", never "mentally disabled". That part made me laugh inside), they apparently have a cold relationship, because all suburban marriages in contemporary cinema must be cold. Their other two kids are Michelle Williams, who is apparently an aspiring actress about to attend college, and Jena Malone, who plays the same troubled teen-archetype she always plays, this time with a heroine addiction. Malone was also the girlfriend of Leland, which gives him his link to his victim. Williams' boyfriend, who was orphaned and came to live with the family, and is a baseball player looking to go to the same college as his girlfriend, is played by Chris Klein. He ends up doing more with his character than any of the other bit players, managing to steal the movie at times. Lena Olin is Leland's mother, who seems to be perpetually sad for some reason. Kevin Spacey (also the executive producer) is Leland's cold and absentee father, who is a famed novelist. Eventually, Sherilyn Fenn will show up to put a wrinkle in Leland's story-- if you even care at that point. Oh yeah, and there's a drug-dealing ex-boyfriend, a couple of fellow juvees, and a co-worker of Pearl's with whom he has an affair on his long distance girlfriend with (played by Kerry Washington).<br /><br />Sorry if all that synopsis and character breakdown took so long. If it seemed meaningless and boring, then you've just experienced a bit of what I did during the 108 minutes I spent watching the movie. But the unruly supporting cast of over-wrought clichés is the least of this film's crimes. The biggest one is that the whole exercise is entirely pointless. We aren't given a fascinating look into a troubled mind, we aren't given an effective explanation, we aren't given much of anything. Given that it sucked so much, I'm gonna go ahead and spoil the ending for you so that you never have to see it: Leland stabbed the retard because all Leland could see in the world was sadness, and wanted to spare Corky (or whatever the victim's name was) the sadness in his eyes. It's like the worst emo band in the world made an album, and titled it "The World Is Sad, So I Killed A Retard". Oh, and Leland dies in the end, in a sequence so reliant of unbelievable coincidences that it would have ruined the movie, if the movie didn't already suck. Of course he dies in the end, because that made the movie so deep.<br /><br />I'm giving the movie 2 stars, because the actors themselves all did a pretty good job with the junk they were given. The scenes with Cheadle and Gosling together were even interesting on some levels. But, to paraphrase the film itself, you have to believe that movies are more than the sum of their parts, kiddo.

- True label: Negative
- Prediction: Positive
- Confidence: 0.87734568
- Error type: TODO(student)
- Hypothesis: TODO(student)

Các nhóm gợi ý để tự kiểm tra: negation, sarcasm, long-range dependency, ambiguous sentiment, domain-specific expression, rare word, tokenization, context, label ambiguity. Không xem keyword là bằng chứng nhân quả.

## Bài 28 — Sáu câu hỏi bắt buộc

### Q1. Model nào hưởng lợi nhiều hơn khi training data ít?

- Observation: TODO(student)
- Evidence: TODO(student): tham chiếu results.csv; chưa có đủ thực nghiệm thì ghi rõ.
- Interpretation: TODO(student)

### Q2. Nếu pretrained model tốt hơn from-scratch, điều đó chứng minh điều gì?

- Observation: TODO(student)
- Evidence: TODO(student): tham chiếu results.csv; chưa có đủ thực nghiệm thì ghi rõ.
- Interpretation: TODO(student)

### Q3. Nếu frozen model gần bằng fine-tuned model, điều đó có ý nghĩa gì?

- Observation: TODO(student)
- Evidence: TODO(student): tham chiếu results.csv; chưa có đủ thực nghiệm thì ghi rõ.
- Interpretation: TODO(student)

### Q4. Nếu fine-tuning tốt hơn frozen model rất nhiều, điều đó gợi ý điều gì?

- Observation: TODO(student)
- Evidence: TODO(student): tham chiếu results.csv; chưa có đủ thực nghiệm thì ghi rõ.
- Interpretation: TODO(student)

### Q5. Nếu performance giảm mạnh trên domain khác, nguyên nhân có thể là gì?

- Observation: TODO(student)
- Evidence: TODO(student): tham chiếu results.csv; chưa có đủ thực nghiệm thì ghi rõ.
- Interpretation: TODO(student)

### Q6. Nếu sequence length tăng nhưng performance không đổi trong khi training time tăng mạnh, có nên tăng tiếp không?

- Observation: TODO(student)
- Evidence: TODO(student): tham chiếu results.csv; chưa có đủ thực nghiệm thì ghi rõ.
- Interpretation: TODO(student)

