# QA
final project during DLS2020 Semester 2

## Yes/No Questions

We will be working with a BoolQ body. The corpus consists of questions assuming a binary answer (yes / no), paragraphs from Wikipedia, the first answer to the question, the title of the article from which the paragraph was extracted and the answer itself (true / false).
The case is described in the article:

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, Kristina Toutanova
BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions - https://arxiv.org/abs/1905.10044

The corpus (train-dev split) is available in the project repository: https://github.com/google-research-datasets/boolean-questions

We use a part of the corpus for training train, for validation and testing - the dev part.

### Question example: 
> question: is batman and robin a sequel to batman forever

> title: Batman & Robin (film)

> answer: true

> passage: With the box office success of Batman Forever in June 1995, Warner Bros. immediately commissioned a sequel. They hired director Joel Schumacher and writer Akiva Goldsman to reprise their duties the following August, and decided it was best to fast track production for a June 1997 target release date, which is a break from the usual 3-year gap between films. Schumacher wanted to homage both the broad camp style of the 1960s television series and the work of Dick Sprang. The storyline of Batman & Robin was conceived by Schumacher and Goldsman during pre-production on A Time to Kill. Portions of Mr. Freeze's back-story were based on the Batman: The Animated Series episode ''Heart of Ice'', written by Paul Dini.

### See code in [notebook](https://colab.research.google.com/drive/12uYSW9LKzgD2iIrMxXB6ve4VzOuQhhig?usp=sharing)

### Results 

***Fasttext:*** 
* *Model 1.* `wordNgrams=2, epoch=15`

Train accuracy: <mark>0.9275</mark>, Dev: <mark>0.6853</mark>

* *Model 2.* `wordNgrams=3, epoch=10, lr=0.2, dim=50`

Train accuracy: <mark>0.9567</mark>, Dev: <mark>0.6945</mark>


***bert_uncased_L-4_H-256_A-4:*** 

| data | best val epoch | accuracy | acc. class 1 | acc. class 2 | f1 score |
| --- | --- | --- | --- | --- | -- |
| val | 9 | 0.626 | 0.515 | 0.686 | 0.705 | 
| test | 9 | 0.649 | 0.520 | 0.713 | 0.730 | 
|  |  |  |  |  |  |
| val | 12 | 0.662 | 0.554 | 0.744 | 0.715 | 
| test | 12 | 0.698 | 0.575 | 0.785 | 0.752 | 
|  |  |  |  |  |  |
| val | 11 | 0.666 | 0.602 | 0.687 | 0.756 | 
| test | 11 | <mark>0.705</mark> | 0.647 | 0.723 | <mark>0.789</mark> | 


***bert_uncased_L-8_H-512_A-8:*** 

| data | best val epoch | accuracy | acc. class 1 | acc. class 2 | f1 score |
| --- | --- | --- | --- | --- | -- |
| val | 8 | 0.675 | 0.572 | 0.746 | 0.730 | 
| test | 8 | 0.705 | 0.587 | 0.782 | 0.762 | 
|  |  |  |  |  |  |
| val | 8 | 0.712 | 0.618 | 0.775 | 0.762 | 
| test | 8 | 0.732 | 0.617 | 0.813 | 0.781 | 
|  |  |  |  |  |  |
| val | 7 | 0.700 | 0.653 | 0.717 | 0.776 | 
| test | 7 | <mark>0.740</mark> | 0.688 | 0.760 | <mark>0.809</mark> | 


***distilroberta-base:*** 

| data | best val epoch | accuracy | acc. class 1 | acc. class 2 | f1 score |
| --- | --- | --- | --- | --- | -- |
| val | 5 | 0.758 | 0.718 | 0.777 | 0.812 | 
| test | 5 | <mark>0.762</mark> | 0.694 | 0.795 | <mark>0.818</mark> | 
|  |  |  |  |  |  |
| val | 7 | 0.702 | 0.628 | 0.739 | 0.767 | 
| test | 7 | 0.720 | 0.630 | 0.763 | 0.786 | 

