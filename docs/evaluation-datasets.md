# Evaluation datasets and automatic metrics for a friend-like emotional-support companion

Research date: 2026-11-16. Facts are marked verified (seen on the primary source, with a quote) or unverified. Licence notes are not legal advice.

## How to read this report

- **[V]** means verified on a primary source in this session: the arXiv or ACL Anthology paper, the official GitHub repo, an official Hugging Face card or the HF datasets-server API, or an official project page. A short verbatim quote is given where it matters.
- **[U]** means unverified. Either the page could not be fetched, the fact came only from a search-engine summary or secondary source, or it is my own inference. Inferences are labelled as such.
- **"LLM-judge"** marks a benchmark whose official scoring uses a general LLM as a judge. **"Model-based scorer"** marks scoring by a fine-tuned, fixed-weight classifier. A fixed classifier is reproducible, but it is still a learned model.
- Licences come from the LICENSE file or HF card metadata. I am not a lawyer. Treat any **NC (non-commercial)** licence as a problem for a project that may become commercial, even for internal evaluation only: whether internal benchmarking of a commercial product counts as "commercial use" is a legal question.

Pages and files I fetched directly: arXiv API abstracts for all arXiv-hosted papers; full-text PDFs (via `pdftotext`) of ED, ESConv, AugESC, ExTES, EmoBench, EQ-Bench, ESC-Eval, MESC, HOPE, PAIR, EPITOME, DailyDialog, Liu et al. 2016, MSC, LoCoMo, LongMemEval, PsyEval, DialogBench, DiaSafety, ProsocialDialog, XSTest, HarmBench, CPsyExam, MentalBench, CounselBench, MISC, CRADLE Bench, and "Between Help and Harm"; GitHub READMEs and LICENSE files; HF dataset cards and the HF API; Zenodo API records; and the UMD suicidality page.

Pages I could **not** fetch:
- EmotionLines site (doraemon.iis.sinica.edu.tw): HTTP 502.
- DailyDialog homepage (yanran.li): the domain is now a parked "for sale" page.
- PsyEval `data-usage-policy.md`: 404.
- `github.com/abachaa/MedQuAD` raw README: 404. I read the repo page through a web fetch instead.

---

## 1. Empathetic / emotional-support response generation

### 1.1 EmpatheticDialogues (ED), Rashkin et al., ACL 2019
- **URLs:** https://arxiv.org/abs/1811.00207 · https://aclanthology.org/P19-1534 · https://github.com/facebookresearch/EmpatheticDialogues · https://huggingface.co/datasets/facebook/empathetic_dialogues
- **Licence [V]:** CC BY-NC 4.0. The repo LICENSE begins "Attribution-NonCommercial 4.0 International", and the HF card says `license: cc-by-nc-4.0`. **Non-commercial.**
- **Size/splits [V]:** "The final train/val/test split was 19533 / 2770 / 2547 conversations" (paper). The HF rows are utterance-level: train 76,673 / validation 12,030 / test 10,943. The paper says "We consider 32 emotion labels". English, US crowdworkers.
- **Standard metrics [V]:** P@1,100 (retrieval), BLEU and perplexity, plus human ratings. The README says "automated metrics (P@1,100 and BLEU) are improved…". Table 1 reports P@1,100, AVG BLEU and PPL.
- **Released scorers [V]:** a fastText emotion classifier (`fasttext_empathetic_dialogues.mdl`, used for EmoPrepend-1). Later work adds classifier metrics on ED (see 1.12): EmoAcc, IntentAcc, EPITOME, diff-EPITOME.
- **Criticism [V]:** the paper itself warns that "automated metrics don't always correlate with human judgments of dialogue quality (Liu et al., 2016)".
- **LLM-judge?** No. The paper uses human evaluation.

### 1.2 ESConv (Emotional Support Conversation), Liu et al., ACL 2021
- **URLs:** https://arxiv.org/abs/2106.01144 · https://aclanthology.org/2021.acl-long.269 · https://github.com/thu-coai/Emotional-Support-Conversation · https://huggingface.co/datasets/thu-coai/esconv
- **Licence [V]:** the LICENSE is CC BY-NC 4.0, and the README says "Data and codes are for academic research use only." **Non-commercial / academic only.**
- **Size [V]:** the README says "ESConv now contains 1,300 conversations with 10 topic problems". The paper's version has 1,053 dialogues. HF datasets-server shows train 910 / validation 195 / test 195. The paper says "We split ESConv into the sets of training / validation / test with the proportions of 6:2:2". English, crowdworkers trained on Helping Skills. There are 8 strategies: Questions, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Reflection of feelings, Information, Restatement or Paraphrasing, Other. These map directly onto your "reflect feelings / ask questions / gentle suggestions" strategies.
- **Standard metrics [V]:**
  - Paper: "The automatic metrics we adopted include perplexity (PPL), BLEU-2 (B-2), ROUGE-L (R-L), and the BOW Embedding-based … Extrema matching score".
  - Follow-up work adds **strategy prediction accuracy**. MISC (ACL 2022) says: "We take the strategy prediction accuracy ACC. as an essential metric… We then acquire the conventional PPL (perplexity), B-2 (BLEU-2), B-4 (BLEU-4), R-L (ROUGE-L) and M (Meteor)". Distinct-n is also widely reported in this line of work [U for the specific papers; the metric is from Li et al. 2016].
- **Released scorers:** none for automatic quality scoring. The 2024 repo update adds `FailedESConv.json` (196 negative samples).
- **Criticism / human eval [V]:** the paper relies on human interactive evaluation (Fluency, Identification, Comforting, and others). The README shows that the post-chat survey recorded seeker emotion-intensity change and empathy/relevance scores. That is an outcome-style signal, but it needs real users.
- **LLM-judge?** No.

### 1.3 AugESC, Zheng et al., Findings ACL 2023
- **URLs:** https://arxiv.org/abs/2202.13047 · https://github.com/thu-coai/AugESC · https://huggingface.co/datasets/thu-coai/augesc
- **Licence [V]:** CC BY-NC 4.0 (HF). **Non-commercial.**
- **Size [V]:** "AugESC contains 65K dialogue sessions and 1,738K utterances". HF has 65,077 train rows. The HF card says it was "distilled from a fine-tuned GPT-J model". There is no strategy annotation [U: inferred from the absence of strategy fields in the card].
- **Metrics [V]:** quality was judged by humans ("Through comprehensive human evaluation"). It is meant as training data, not a test benchmark.
- **LLM-judge?** No, but the data is itself LLM-generated.

### 1.4 ExTES, Zheng et al., ACL 2024 ("Building Emotional Support Chatbots in the Era of LLMs")
- **URLs:** https://arxiv.org/abs/2308.11584 · https://aclanthology.org/2024.acl-long.611 · https://github.com/pandazzh2020/ExTES
- **Licence [V]:** **none stated.** The repo has no LICENSE file and the GitHub sidebar shows no licence. A third-party catalogue claims CC BY 4.0 [U]. The data was generated with ChatGPT, which may raise provider-terms questions [U, inference].
- **Size [V]:** "a total of 11,177 dialogues". The README lists 16 strategy categories (Reflective Statements, Emotional Validation, Affirmation, Offer Hope, Suggest Options, …) and 36 scenarios. English.
- **Metrics [V]:** "METEOR, BLEU-4 (B-4), ROUGE-L (R-L), Vector Extrema and the Distinct2/3", plus human evaluation.
- **LLM-judge?** No, but the data is LLM-generated.

### 1.5 PsyQA, Sun et al., Findings ACL 2021
- **URLs:** https://arxiv.org/abs/2106.01702 · https://github.com/thu-coai/PsyQA
- **Licence/access [V]:** **Chinese only.** You need a signed user agreement emailed to the authors; the repo hosts only a sample file.
- **Size [V]:** "22K questions and 56K long and well-structured answers".
- **Metrics [V]:** generation metrics via the NLG-evaluation toolkit (README), plus human metrics. PsyEval later reuses its human criteria.
- **Fit:** low. It is Chinese and gated.

### 1.6 MESC (Multimodal Emotional Support Conversation), Chu et al., arXiv 2408.03650 (IEEE TMM 2025 [U])
- **URLs:** https://arxiv.org/abs/2408.03650 · https://github.com/chuyq/MESC
- **Licence [V]:** no LICENSE file. The source is the TV show *In Treatment*: "The MESC dataset is sourced from the TV show *In Treatment*, specifically from seasons 1 to 3." Copyrighted source material.
- **Size [V]:** "28,762 utterances, 1,019 dialogues", annotated with 7 emotions and 10 therapeutic strategies.
- **Metrics [V]:** "Accuracy and Weighted F1 … for emotion recognition, strategy prediction, and system emotion prediction … BLEU-n (B-2), ROUGE-L (R-L), and BERTScore".
- **Fit:** low because of the licence and the scripted-TV domain.

### 1.7 Counsel-Chat (CounselChat), Bertagnolli 2020
- **URLs:** https://github.com/nbertagnolli/counsel-chat · https://huggingface.co/datasets/nbertagnolli/counsel-chat
- **Licence [V]:** the repo LICENSE is MIT and the HF card says "Licensing Information: MIT". However, the data is "Scrape of Counselchat.com's forum", "written by licensed counselors in the United States and annonymous individuals". The MIT label on scraped third-party content is legally shaky [U, inference].
- **Size [V]:** 2,775 rows (HF). English, US therapists. There are no official splits on the card ("[More Information Needed]").
- **Metrics:** no official benchmark metric. Upvotes are available as a weak quality signal [V: an `upvotes` field exists]. CounselBench (§5.3) builds expert evaluations on it.

### 1.8 HOPE (Mental Health cOunselling of PatiEnts), Malhotra et al., WSDM 2022 (IIIT-Delhi)
- **URLs:** https://arxiv.org/abs/2111.06647 · https://github.com/LCS2-IIITD/SPARTA_WSDM2022
- **Access [V]:** "To request access to the HOPE Dataset … Fill out the agreement form" and email the authors. No licence is stated.
- **Size [V]:** "∼ 12.9K utterances across 212 mental-health counseling sessions", taken from "publicly-available counselling session videos on YouTube", with 12 dialogue-act labels.
- **Metric [V]:** dialogue-act classification: "we compute macro-F1, weighted-F1, and accuracy scores".
- **Fit:** useful for a "which counselling act is this?" classifier, not for scoring replies.

### 1.9 AnnoMI (Motivational Interviewing), Wu et al., ICASSP 2022 / Future Internet 2023
- **URLs:** https://github.com/uccollab/AnnoMI
- **Licence [V]:** **no licence file or sidebar licence.** The extended paper abstract says "the first publicly and freely available" dataset.
- **Size [V]:** "133 faithfully transcribed and expert-annotated demonstrations of high- and low-quality motivational interviewing". There are utterance-level labels: therapist behaviour (reflection/question/therapist_input/other), reflection simple/complex, question open/closed, client talk type, and conversation-level `mi_quality`.
- **Metrics [U]:** utterance-level classification baselines (the paper reports these; specific metrics not checked).
- **Fit:** you could train a "reflection vs question vs advice" tagger on it to measure how your bot's strategy distribution compares with high-quality MI. That would be an automatic, classifier-based strategy-mix metric. Check the licence first.

### 1.10 EPITOME / "Empathy in text-based mental health support", Sharma et al., EMNLP 2020
- **URLs:** https://arxiv.org/abs/2009.08441 · https://github.com/behavioral-data/Empathy-Mental-Health
- **What it is [V]:** three empathy mechanisms, "Emotional Reactions, Interpretations, and Explorations", each labelled "no communication, weak communication, and strong communication". The corpus has "10k (post, response) pairs". Only the **Reddit portion** is in the repo; I counted 3,084 pairs per mechanism CSV. The TalkLife portion is "for non-commercial use" via TalkLife.
- **Classifier [V]:** a RoBERTa bi-encoder with "∼80% acc., ∼70% macrof1". The official repo **does not release trained weights**: Releases is empty and the README trains to `output/*.pth`.
- **Licences [V]:** two .docx licence files in the repo.
  - The "academic license" for the **Model**: "Use of the Model is restricted to non-commercial research purposes … For questions about commercial use, please contact Tim Althoff".
  - A BSD-style "attribution only license" for the **Software**, which requires a UW acknowledgement "on splash pages on start-up of your research software".
  - The Reddit data's own licence is not stated.
- **Third-party checkpoints [V]:**
  - EmpGPT-3 (Lee et al., COLING 2022) says: "We provide all model checkpoints used in our work through google drive … Epitome, EmoAcc, IntentAcc, and diff-Epitome".
  - The focused-empathy repo (Kim et al., EMNLP 2021) "will automatically download the RoBERTa models finetuned on EmpatheticDialogues" for Interpretation and Exploration scores.
  - HF has community uploads (e.g., `RyanDDD/empathy-mental-health-reddit-ER/IP/EX`) of unknown provenance [U].
  - All derive from the NC-licensed model or data.
- **Use as metric [V]:** PsyEval "integrated evaluation metrics proposed by EPITOME" (§6.2). Lee et al. 2022 propose EPITOME / diff-EPITOME as an automatic empathy metric for ED.
- **Fit:** this is the best-established **classifier-based empathy metric**, but it is **non-commercial**. It is also known mainly from Reddit peer-support text [U: whether it transfers to LLM-style replies is not established].

### 1.11 PAIR (counsellor reflection scoring), Min et al., EMNLP 2022
- **URLs:** https://aclanthology.org/2022.emnlp-main.11
- **What [V]:** "takes as input one dialog turn containing a client prompt … and a counselor response … and outputs a numeric score indicating the quality of the reflection". The paper evaluates with "recall@1, Pearson and Spearman, and Kendall's Tau".
- **Release [U]:** no official code or data repo was found. HF has third-party re-implementations (`Khriis/PAIR`, `coding-zzz-oe/PAIR`), unverified. The dataset licence is unknown.
- **Fit:** conceptually ideal as a "reflection quality" scorer, but not reproducibly available.

### 1.12 Classifier-based empathy metrics for ED (Lee, Lim & Choi, COLING 2022)
- **URLs:** https://aclanthology.org/2022.coling-1.56 · https://github.com/passing2961/EmpGPT-3
- **What [V]:** automatic metrics EmoAcc (emotion classifier accuracy against the ED label), IntentAcc (empathetic-intent classifier, from Welivita et al.'s EmpatheticIntents), EPITOME and diff-EPITOME. The checkpoints are on Google Drive. The repo is MIT, but the underlying EPITOME model licence is NC (see 1.10).

### 1.13 EmoBench, Sabour et al., ACL 2024 ⭐
- **URLs:** https://arxiv.org/abs/2402.12071 · https://aclanthology.org/2024.acl-long.326 · https://github.com/Sahandfer/EmoBench · https://huggingface.co/datasets/SahandSab/EmoBench
- **Licence [V]:** MIT for both repo and HF. **Commercial-friendly.**
- **Size [V]:** "400 hand-crafted scenarios in English and Chinese". It has two tasks: **Emotional Understanding** (emotion and cause) and **Emotional Application** ("identify the most effective response or action"). HF datasets-server shows 400 rows in each of the `emotional_understanding` and `emotional_application` configs.
- **Scoring [V]:** MCQ. "Our tasks take the form of multiple-choice questions (MCQ)". The paper permutes the option order ("4 permutations") and reports "the average accuracy of the four runs". It has a human baseline ("the best-performing LLM (GPT-4) falling short of the average human's performance").
- **LLM-judge?** No. It uses exact-match accuracy.
- **Criticism:** small, partly Chinese, and it tests EI reasoning, not reply quality [inference].

### 1.14 EQ-Bench, Paech, arXiv 2312.06281 ⭐ (legacy v2 only)
- **URLs:** https://arxiv.org/abs/2312.06281 · https://github.com/EQ-bench/EQ-Bench · https://eqbench.com
- **Licence [V]:** the repo is MIT.
- **What and how [V]:** the model predicts "the intensity of emotional states of characters in a dialogue". From the README: "V2 of EQ-Bench contains 171 questions", "we scale down smaller differences (differences between 1-4 from reference) on a curve", and "a score of 0 corresponds to answering randomly, and a score of 100 matches the reference answers exactly". The eqbench.com about page says legacy EQ-Bench "is graded without the need for a judge".
- **Criticisms [V]:**
  - "EQ-Bench correlates strongly with … MMLU (r=0.97)". It may largely measure general capability.
  - "All questions and reference answers were determined by the authors of this paper". The reference answers are subjective.
  - The dialogues were generated by GPT-4.
  - The README notes run-to-run variance from sampler and quantisation perturbations.
- **⚠ LLM-judge [V]:** **EQ-Bench 3** (the current leaderboard) "is a LLM-judged test judged by Claude Opus 4.6". The repo's Creative Writing and Judgemark tasks are also LLM-judged. **Use only legacy v2.**

### 1.15 ESC-Eval, Zhao et al., EMNLP 2024 — ⚠ model-based / human
- **URLs:** https://arxiv.org/abs/2406.14952 · https://github.com/AIFlames/Esc-Eval
- **How [V]:** a role-play agent (ESC-Role) chats with your model, followed by human evaluation in "7 dimensions (i.e., fluency, diversity, empathy, information, humanoid, skillful, and overall)". For automation, "ESC-RANK, which trained on the annotated data" scores the dialogues. ESC-RANK is a fine-tuned InternLM2-chat-7B (README) — a **fine-tuned LLM scorer**, and the user simulator is also an LLM.
- **Licence:** no LICENSE file was found in the repo [V: absent]. 2,801 role cards are re-organised from 7 existing datasets, which inherit their licences [U].
- **Fit:** not "automatic reference-based". Flag as LLM-in-the-loop.

### 1.16 CPsyCoun / CPsyCounE, Zhang et al., Findings ACL 2024 — ⚠ LLM-judge
- **URLs:** https://arxiv.org/abs/2405.16433 · https://github.com/CAS-SIAT-XinHai/CPsyCoun
- **Licence [V]:** the repo LICENSE is CC BY 4.0. The HF `CAS-SIAT-XinHai/CPsyCoun` card says cc-by-sa-4.0. Chinese.
- **Scoring [V]:** "we employ LLM to assess these responses, utilizing the evaluation metrics" (Comprehensiveness, Professionalism, Authenticity, Safety). CPsyCounE is "the general multi-turn dialogue evaluation dataset, which has nine topics". **This is LLM-as-judge. Exclude.**

---

## 2. Emotion recognition / understanding as a component

### 2.1 GoEmotions, Demszky et al., ACL 2020 ⭐
- **URLs:** https://arxiv.org/abs/2005.00547 · https://github.com/google-research/google-research/tree/master/goemotions · https://huggingface.co/datasets/google-research-datasets/go_emotions
- **Licence [V]:** HF card says "Apache License 2.0". **Commercial-friendly.**
- **Size [V]:** "58k carefully curated comments extracted from Reddit … 27 emotion categories or Neutral". The filtered splits are "43,410 training examples, 5426 dev examples and 5427 test examples". English.
- **Metric [V]:** multi-label F1. "Our BERT-based model achieves an average F1-score of .46".
- **India note [V]:** README: "annotators were all native English speakers from India".
- **Released classifier:** Google provides baseline code and a tutorial, not official weights [V: README points to code and tutorial]. A popular third-party model, `SamLowe/roberta-base-go_emotions`, is MIT-tagged on HF [V: tag] with unverified provenance and quality.
- **Fit:** (a) test your emotion-detection component; (b) use a GoEmotions classifier as an automatic "does the reply acknowledge the user's emotion?" proxy. The proxy use is non-standard; label it as your own metric.

### 2.2 EmotionLines, Hsu et al., LREC 2018
- **URLs:** https://arxiv.org/abs/1802.08379 · official site unreachable (HTTP 502)
- **Size [V]:** "29,245 utterances from 2,000 dialogues … one of seven emotions, six Ekman's basic emotions plus the neutral emotion". Sources are Friends scripts and private Facebook Messenger dialogues.
- **Licence/access [U]:** could not verify.

### 2.3 MELD, Poria et al., ACL 2019
- **URLs:** https://arxiv.org/abs/1810.02508 · https://github.com/declare-lab/MELD · https://affective-meld.github.io
- **Licence [V]:** the repo LICENSE is **GPL-3.0**. The content comes from the *Friends* TV series.
- **Size [V]:** 1,039 / 114 / 280 dialogues and 9,989 / 1,109 / 2,610 utterances (train/dev/test). Labels are 7 emotions (Anger, Disgust, Sadness, Joy, Neutral, Surprise, Fear) plus sentiment. Pretrained baseline weights are downloadable.
- **Metric [U]:** weighted F1 is the commonly reported metric. I did not verify it on the primary page.

### 2.4 DailyDialog, Li et al., IJCNLP 2017
- **URLs:** https://arxiv.org/abs/1710.03957 · https://aclanthology.org/I17-1099 · https://huggingface.co/datasets/li2017dailydialog/daily_dialog (the homepage yanran.li is now a parked domain)
- **Licence [V]:** the HF card says "DailyDialog dataset is licensed under CC BY-NC-SA 4.0". **Non-commercial.**
- **Size [V]:** "13,118 multi-turn dialogues". HF splits are 11,118 / 1,000 / 1,000 dialogues. The emotion labels use Ekman's "BigSix Theory" plus other/neutral. English-learner-style everyday dialogue.

### 2.5 DialogBench, Ou et al., NAACL 2024
- **URLs:** https://arxiv.org/abs/2311.01677 · https://github.com/kwai/DialogBench
- **Licence [V]:** README badges show "Data License-MIT" and "Code License-MIT".
- **What [V]:** 12 tasks, including **Emotion Detection (823 items)**, Personality-grounded Response Generation (832), Offensive Detection (802), Multi-turn Response Generation (800), and Dialogue NLI. The paper says "adopt multi-choice questions and use accuracy as the evaluation metric". English and Chinese.
- **Caveats [V]:** "we prompt GPT-4 to generate evaluation instances". The paper also shows LLM position bias in option selection.
- **LLM-judge?** No. Scoring is exact-match MCQ, although the items themselves were LLM-generated.

### 2.6 EmoBench and EQ-Bench (legacy)
See 1.13 and 1.14. Both are judge-free emotion-understanding benchmarks.

### 2.7 IndieMH (Hinglish counselling emotion recognition), LREC 2026 — India-relevant
- **URL:** https://aclanthology.org/2026.lrec-1.226/
- **What [V, abstract]:** "Hinglish, a code-mixed blend of Hindi and English … annotate IndieMH with 13 emotional states". Search results report accuracy and weighted-F1 metrics [U].
- **Access/licence [U]:** the Anthology page has no dataset or code link.

---

## 3. Safety and crisis handling

### 3.1 Reddit C-SSRS Suicide Dataset, Gaur et al., WWW 2019 ⭐ (open)
- **URLs:** https://zenodo.org/records/2667859 (v1) · https://zenodo.org/records/4543776 (v2, 448 users with post-level labels) · https://github.com/manasgaur/Knowledge-aware-Assessment-of-Severity-of-Suicide-Risk-for-Early-Intervention
- **Licence/access [V]:** Zenodo API returns `access_right: open` and `license: cc-by-4.0` for both records. **No DUA.** It is Reddit-derived, so Reddit's terms and ethics apply [inference].
- **Size [V]:** the README says "500 Redditors (anonymized), their posts and domain expert annotated labels". The file `500_Reddit_users_posts_labels.csv` is 3.6 MB, alongside lexicon CSVs. The label scheme is C-SSRS-based (Supportive / Indicator / Ideation / Behavior / Attempt) [U: label names from the search summary and v2 description; the v2 description lists "Supportive, Suicide Ideation, Suicide Behavior, Suicide Attempt"].
- **Metric:** classification (precision/recall/F1). Papers report graded-recall-style variants [U].
- **Fit:** tests the **risk-detection component** on user text, not reply quality. It is small and user-level.

### 3.2 UMD Reddit Suicidality Dataset (CLPsych 2019 shared task, Shing et al. 2018; Zirikly et al. 2019)
- **URL:** https://psresnik.github.io/umd_reddit_suicidality_dataset.html (redirected from umiacs.umd.edu)
- **Access [V]:** an application to Philip Resnik, reviewed by a governance committee. You must submit IRB protocol and approval: "your application will not be reviewed without" them. The page does not state commercial terms.
- **Size/labels [V]:** 11,129 SuicideWatch users plus 11,129 controls, an expert-annotated subset of 245+245, and a crowd subset of 621+621. Risk is rated "on a four-point scale of no risk, low, moderate, and severe risk".
- **Fit:** the gold-standard research benchmark, but **effectively unavailable to a side project** because of IRB and committee approval. CLPsych 2024 used a strict-use DUA via the American Association of Suicidology [U, from secondary search].

### 3.3 ProsocialDialog, Kim et al., EMNLP 2022 ⭐
- **URLs:** https://arxiv.org/abs/2205.12688 · https://github.com/skywalker023/prosocial-dialog · https://huggingface.co/datasets/allenai/prosocial-dialog
- **Licence [V]:** data is CC BY 4.0 (HF) and the repo is MIT. **Commercial-friendly.**
- **Size [V]:** "58K dialogues, with 331K utterances, 160K unique RoTs, and 497K dialogue safety labels". HF rows are train 120,236 / validation 20,416 / test 25,029. English. The unsafe user turns were written by GPT-3 and the prosocial replies by crowdworkers.
- **Labels [V]:** `safety_label` ∈ {casual, possibly_needs_caution, probably_needs_caution, needs_caution, needs_intervention}.
- **Released model [V]:** Canary (safety and rules-of-thumb generator): "You can now directly download our Canary". The README points to COSMO instead of Prost.
- **Fit:** classify each user turn into the 5 safety levels (macro-F1). This is directly relevant to "when do I escalate to crisis mode?". Response-side scoring would need Canary or a classifier [inference].

### 3.4 DiaSafety, Sun et al., Findings ACL 2022
- **URLs:** https://arxiv.org/abs/2110.08466 · https://github.com/thu-coai/DiaSafety
- **Licence [V]:** the repo LICENSE is Apache-2.0, and the README says "We have uploaded the our dataset in this repository".
- **What [V]:** context-sensitive unsafe *bot responses*: "(1) Offending User, (2) Risk Ignorance, (3) Unauthorized Expertise, (4) Toxicity Agreement, (5) Biased Opinion, and (6) Sensitive Topic Continuation". **Risk Ignorance** (ignoring a user's self-harm signals) and **Unauthorized Expertise** (medical advice) map directly onto a companion bot's risks. The authors "train a dialogue safety classifier".
- **Released classifier [U]:** the README does not link checkpoints and I found none on HF. You may have to train your own on the released data.
- **Fit:** strong. It scores **your bot's replies** in context with a classifier. English.

### 3.5 SafetyBench, Zhang et al., ACL 2024
- **URLs:** https://arxiv.org/abs/2309.07045 · https://github.com/thu-coai/SafetyBench · https://huggingface.co/datasets/thu-coai/SafetyBench
- **Licence [V]:** MIT.
- **What [V]:** "11,435 diverse multiple choice questions spanning across 7 distinct categories". HF has `test/en` with 11,435 rows, plus zh and a zh_subset. Since "2025/07/28 … we've decided to fully open-source our test answers", local exact-match scoring is possible.
- **Fit:** general safety *knowledge*, not crisis replies. MCQ accuracy.

### 3.6 XSTest, Röttger et al., NAACL 2024 ⭐ (over-refusal)
- **URLs:** https://arxiv.org/abs/2308.01263 · https://github.com/paul-rottger/xstest · https://huggingface.co/datasets/Paul/XSTest
- **Licence [V]:** "XSTest prompts are subject to … CC-BY-4.0".
- **What [V]:** "250 safe prompts … and 200 unsafe prompts as contrasts" (450 rows on HF).
- **Scoring [V]:** the paper hand-labelled responses ("we evaluate all models by manually annotating"). The repo provides `classify_completions_strmatch.py`, a **pure string-prefix refusal detector** (e.g. "i'm sorry", "i cannot", "as an ai" → `2_full_refusal`), and `classify_completions_gpt.py`, which is a GPT judge.
- **Fit:** a companion must not refuse "I could kill for a coffee" or lecture the user. Use the **string-match** classifier for fully automatic scoring, accepting its crudeness [inference].

### 3.7 HarmBench, Mazeika et al., 2024 — model-based scorer
- **URLs:** https://arxiv.org/abs/2402.04249 · https://github.com/centerforaisafety/HarmBench
- **Licence [V]:** the repo is MIT; HF `walledai/HarmBench` is MIT and gated (auto).
- **Scoring [V]:** attack success rate (ASR) via released classifiers such as "cais/HarmBench-Llama-2-13b-cls". This is a fixed, fine-tuned LLM classifier. The Llama-2 licence terms of that classifier were not checked [U].
- **Fit:** jailbreak/misuse robustness. Lower priority for a companion.

### 3.8 Newer mental-health crisis benchmarks (2025–2026)
- **CRADLE Bench** (arXiv 2510.23845) [V abstract/paper]:
  - "600 clinician-annotated evaluation examples and 420 development examples" across seven crisis types (suicidal ideation, rape, domestic violence, child abuse, sexual harassment, …) with temporal labels.
  - Metrics are "Exact Match, Jaccard, Micro/Macro F1, Recall", which are automatic.
  - The paper says "All data, models, and code are released", but I found no URL in the text. Licence unverified. Reddit-derived. The training set is LLM-ensemble-labelled.
  - **Promising for the detection component; verify the licence.**
- **Between Help and Harm** (arXiv 2509.24857, JMIR Mental Health) [V]: 2,252 crisis inputs from 12 datasets in a 6-category taxonomy. Both crisis categorisation and response appropriateness use "the LLM-as-a-judge technique". **⚠ LLM-judge.** Code at https://github.com/ellisalicante/LLMs-Mental-Health-Crisis, with an HF collection.
- **PsyCrisis-Bench** (arXiv 2508.08236) [U, search summary only]: Chinese, LLM-as-judge. **⚠ Exclude.**

---

## 4. Long-term memory in dialogue

### 4.1 Multi-Session Chat (MSC), Xu et al., ACL 2022 ("Beyond Goldfish Memory")
- **URLs:** https://arxiv.org/abs/2107.07567 · https://aclanthology.org/2022.acl-long.356 · https://github.com/facebookresearch/ParlAI/tree/main/projects/msc
- **Licence [V/U]:** ParlAI code is MIT [V]. A separate data licence is **not stated** in the MSC README [U].
- **Size [V]:** "session 1-4, with 237k training examples and 25k valid examples"; session 5 adds roughly 6k validation examples. English crowdworkers.
- **Metric [V]:** **perplexity** per session (Tables 3–5), plus human evaluation. There is no QA-style memory test.
- **Fit:** weak for automatic memory scoring. Perplexity needs model log-probs and doesn't test recall directly [inference].

### 4.2 LoCoMo, Maharana et al., ACL 2024
- **URLs:** https://arxiv.org/abs/2402.17753 · https://github.com/snap-research/locomo · https://snap-research.github.io/locomo/
- **Licence [V]:** "The LOCOMO dataset will be released under the CC BY-NC 4.0 DEED license", and the repo `LICENSE.txt` is CC BY-NC 4.0. **Non-commercial.**
- **Size [V]:** "The benchmark consists of ten conversations" in the current release ("a subset of … 50 conversations"). The conversations were LLM-agent-generated and human-edited, with about 300 turns and 9K tokens on average over up to 35 sessions.
- **Metrics [V]:** QA uses "the F1 partial match metric" after normalisation, which is **automatic**, plus evidence-retrieval accuracy. Event summarisation uses FactScore, which is LLM-based [U on the exact model].
- **Fit:** good QA-style automatic scoring, but NC.

### 4.3 LongMemEval, Wu et al., ICLR 2025 ⭐ (retrieval metrics only, if no LLM judge)
- **URLs:** https://arxiv.org/abs/2410.10813 · https://github.com/xiaowu0162/LongMemEval · https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
- **Licence [V]:** MIT for both repo and HF.
- **Size [V]:** "500 meticulously curated questions" covering "information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention". There are three variants: `_S` (~115k tokens, ~40 sessions), `_M` (~500 sessions), and `_oracle`. The data was cleaned again in 2025-09. User–assistant chat format, English.
- **⚠ Official QA scoring is an LLM judge [V]:** "we prompt-engineer the gpt-4o-2024-08-06 model … the evaluator achieves more than 97% agreement with human experts", and `evaluate_qa.py gpt-4o …`.
- **Automatic part [V]:** "We report Recall@k and NDCG@k" for memory retrieval, using human-annotated evidence sessions and turns (`has_answer: true`).
- **Fit:** the closest match to "remember user facts across sessions". Use Recall@k and NDCG@k (judge-free) and, as a *non-official* approximation, normalised exact or contains-match for short answers plus string-based abstention detection [inference; not the benchmark's protocol].

### 4.4 DialogBench
See 2.5. It is not a memory benchmark, but Personality-grounded Response Generation and Dialogue NLI are MCQ consistency checks.

---

## 5. Grounded / knowledge QA relevant to wellbeing

### 5.1 MedQuAD, Ben Abacha & Demner-Fushman, BMC Bioinformatics 2019
- **URLs:** https://arxiv.org/abs/1901.08079 · https://github.com/abachaa/MedQuAD
- **Licence [V]:** "Creative Commons Attribution 4.0 International Licence (CC BY)" (`LICENSE.txt` too).
- **Size [V]:** "47,457 medical question-answer pairs created from 12 NIH websites". However, "We removed the answers from 3 subsets to respect the MedlinePlus copyright".
- **Evaluation [V]:** a TREC-2017 LiveQA medical test set with "2,479 judged answers", rated "1-Incorrect, 2-Related, 3-Incomplete, and 4-Excellent". This suits retrieval-style scoring (e.g. precision or MAP of retrieved judged answers) [inference on exact metric].
- **Fit:** grounded health-information retrieval with authoritative sources and citations to NIH pages. Its mental-health coverage is limited [U].

### 5.2 MentalQA, Alhuzali et al., arXiv 2405.12619
- **What [V]:** "a novel Arabic dataset featuring conversational-style question-and-answer". **Arabic only. Not a fit.** Licence not checked.

### 5.3 CounselBench, arXiv 2506.08584 — ⚠ human/LLM-judge
- **URLs:** https://github.com/llm-eval-mental-health/CounselBench · https://huggingface.co/datasets/izi-ano/CounselBench-Eval
- **Licence [V]:** **CC BY-NC-ND 4.0.**
- **What [V]:** "2,000 expert evaluations … on patient questions from the public forum CounselChat", plus CounselBench-Adv with "120 expert-authored mental health questions". Scoring needs human experts or LLM judges; the repo has an `llm_as_judges/` folder.
- **Useful finding [V]:** "LLM judges systematically overrate model responses and overlook safety concerns identified by human experts". This is evidence for the owner's no-LLM-judge stance.

### 5.4 MMLU psychology / clinical subsets (Hendrycks et al.)
- **URL:** https://huggingface.co/datasets/cais/mmlu
- **Licence [V]:** MIT.
- **Size [V]:** `professional_psychology` has a 612-question test split. `high_school_psychology`, `clinical_knowledge`, `college_medicine` and `professional_medicine` also exist. MCQ exact match.
- **Fit:** a cheap, standard knowledge sanity check (US-centric).

---

## 6. Psychological counselling skill / knowledge MCQ benchmarks

### 6.1 CPsyExam, Zhao et al., COLING 2025
- **URLs:** https://arxiv.org/abs/2405.10212 · https://github.com/CAS-SIAT-XinHai/CPsyExam · https://huggingface.co/datasets/CAS-SIAT-XinHai/CPsyExam
- **Licence [V]:** HF says cc-by-4.0. **Chinese.**
- **What [V]:** "From the pool of 22k questions, we utilize 4k to create the benchmark". It has single-choice (SCQ) and multiple-answer (MAQ) questions from Chinese exams, including the Psychological Counselor Examination. HF datasets-server shows a 14,854-row `test` split, which does not match the "4k" in the paper [V: both numbers seen; the discrepancy is unexplained]. Scoring is MCQ accuracy.
- **Fit:** low (Chinese).

### 6.2 PsyEval, Jin et al., arXiv 2311.09189
- **URLs:** https://github.com/KaguraRuri/Psy-Eval
- **What [V]:**
  - **USMLE-mental**: 727 MCQs filtered from MedQA.
  - **Crisis Response QA**: 153 questions from IACP and NAMI crisis manuals, "followed by GPT-4 generating three fake answers for each question" and reviewed by medical students.
  - Diagnosis and classification tasks (SMHD, D4).
  - A support-generation task built on PsyQA.
  - "For QA task, accuracy is a suitable metric … For classification task, we also use accuracy". Generation uses BLEU and Distinct-1/2 plus human, GPT-4 and EPITOME criteria.
- **Licence [U]:** the repo points to a `data-usage-policy.md` that returned 404. External datasets keep their own terms (SMHD needs a DUA [U]).
- **Fit:** **Crisis Response QA (MCQ, English) is a useful automatic crisis-knowledge check**, if its licence permits.

### 6.3 MentalBench (DSM-grounded diagnosis), Song et al., arXiv 2602.12871 (2026)
- **URLs:** https://github.com/HoyunS/MentalBench · https://huggingface.co/datasets/hysong/MentalBench
- **Licence [V]:** CC BY 4.0 (HF and README badge).
- **What [V]:** "24,750 synthetic clinical cases" over 23 DSM-5 disorders, in four types (single- and multi-answer). "We measure accuracy using an exact match metric". English.
- **Fit:** a companion should *not* diagnose. This is useful only as a knowledge or calibration probe, or to check the bot refuses to diagnose [inference].
- **Name clash [U]:** "MentalBench-100k" (arXiv 2510.19032, EACL 2026) is a different dataset for evaluating LLM judges.

---

## 7. Metric criticisms (primary sources)

- **Liu et al. 2016, "How NOT to evaluate your dialogue system" (EMNLP) [V]:** word-overlap metrics "correlate very weakly with human judgements in the non-technical Twitter domain, and not at all in the technical Ubuntu domain."
- **Rashkin et al. 2019 [V]:** "automated metrics don't always correlate with human judgments of dialogue quality".
- **BLEURT (Sellam et al. 2020) [V]:** "the most popular choices (e.g., BLEU and ROUGE) may correlate poorly with human judgments". BLEURT and BERTScore are learned or embedding-based alternatives, but they are still reference-based.
- **Distinct-n (Li et al. 2016) [V]:** motivated by "safe, commonplace responses (e.g., 'I don't know')". It measures diversity only, not appropriateness [inference].
- **EQ-Bench [V]:** r=0.97 with MMLU, so it may proxy general intelligence.
- **DialogBench [V]:** LLMs show answer-position bias in MCQ. Permute the options, as EmoBench does.
- **CounselBench [V]:** LLM judges overrate responses and miss expert-flagged safety issues.
- **One-reference problem [inference]:** ESConv and ED have one reference per context, while a supportive reply has many valid forms. Reference-overlap scores penalise good but different replies. ExTES's "Diverse Response Inpainting" (multiple references) partly addresses this [V: ExTES describes DRI as "generating multiple distinct responses for each conversation context"].

---

## 8. Recommendation

### Shortlist (prioritising automatic scoring, commercial-compatible licences, and English friend-like relevance)

| # | Dataset | Capability | Automatic metric | Licence | Why |
|---|---|---|---|---|---|
| 1 | **EmoBench** (EU + EA, English half) | Emotion understanding and **choosing the best supportive response or action** | MCQ accuracy averaged over 4 option permutations | MIT | Judge-free; EA is the closest public proxy for "pick the right support strategy". |
| 2 | **EQ-Bench legacy v2** (171 Q) | Emotion-intensity understanding | Distance-to-reference score, 0 = random, 100 = reference | MIT | Judge-free and cheap. Do **not** use EQ-Bench 3 (Claude-judged). Caveat: r=0.97 with MMLU. |
| 3 | **GoEmotions** | Emotion recognition component; also a classifier you can reuse as a reply-side "emotion acknowledged?" proxy | Multi-label macro-F1 | Apache-2.0 | Large, English, 28 labels; annotators were native English speakers from India. |
| 4 | **ProsocialDialog** (+ **DiaSafety**) | Detect when a user turn needs caution or intervention (Prosocial); detect unsafe bot replies such as *Risk Ignorance* and *Unauthorized Expertise* (DiaSafety) | Macro-F1 of a classifier on held-out test; then % of your replies flagged unsafe by a classifier trained on DiaSafety | CC BY 4.0 / Apache-2.0 | Both permissive; DiaSafety's categories match a companion's actual failure modes. |
| 5 | **Reddit C-SSRS (Zenodo)** (+ **CRADLE Bench** if its licence checks out) | Suicide-risk severity / crisis-type detection on user text | Macro-F1 / graded recall (CRADLE: exact match, Jaccard, micro/macro-F1) | CC BY 4.0, open (CRADLE: unverified) | The only openly licensed, clinician-labelled suicide set found. UMD/CLPsych needs IRB and committee approval. |
| 6 | **XSTest** | Over-refusal / preachiness on safe-but-scary-sounding prompts (plus 200 unsafe contrasts) | % full refusal via the repo's **string-match** classifier | CC BY 4.0 | Companions that refuse "how do I kill a Python process" feel cold; fully automatic, if crude. |
| 7 | **LongMemEval** (`_S` and `_oracle`) | Remembering user facts across sessions, updates, temporal reasoning, abstention | **Recall@k / NDCG@k** on evidence sessions (judge-free). Official answer accuracy uses a GPT-4o judge; a normalised exact/contains match is a non-official substitute | MIT | Closest to "remember user facts". LoCoMo (F1, automatic) is better on scoring but **CC BY-NC**. |
| 8 | **MMLU psychology subsets** + **PsyEval Crisis-Response QA** (licence TBC) + **MedQuAD** | Grounded wellbeing and crisis knowledge | MCQ accuracy; retrieval precision against MedQuAD/LiveQA judged answers | MIT / unverified / CC BY 4.0 | Cheap knowledge-correctness checks. MedQuAD gives citable NIH source URLs. |

**Research-only benchmarks to keep in mind but not to rely on commercially:** ESConv (CC BY-NC, academic only) is the de facto benchmark for **strategy prediction accuracy** plus BLEU-2/ROUGE-L/Distinct. EPITOME empathy classifiers have an NC model licence. ED (CC BY-NC) and DailyDialog (CC BY-NC-SA). If the project stays non-commercial, ESConv strategy accuracy plus EPITOME (ER/IP/EX) scores on ED/ESConv test contexts are the most "standard" reply-level automatic suite. Get a written opinion before using them once it becomes commercial.

**Avoid (LLM-judged or not judge-free):** EQ-Bench 3, CPsyCounE, ESC-Eval (LLM simulator plus fine-tuned LLM ranker), CounselBench (human/LLM judges, and NC-ND), Between Help and Harm, PsyCrisis-Bench, and LongMemEval's official QA scorer.

**Language note:** apart from GoEmotions' Indian annotators and the (unverified-availability) Hinglish IndieMH, none of these benchmarks reflect Indian English, Hinglish, or Indian help-seeking norms and crisis resources. HOPE comes from an Indian lab but uses YouTube counselling videos. Plan a small in-house Indian test set.

### Qualities with no good established automatic metric
- **Warmth, friendliness, "feels like a friend", naturalness/humanlikeness, appropriate humour, non-preachiness, and long-term relationship quality.** No published, validated, judge-free metric exists. BLEU, ROUGE and BERTScore against a single reference correlate weakly with human judgement in open dialogue (Liu et al. 2016) and penalise valid alternative phrasings.
- **What the literature uses instead (verified examples):**
  - Human ratings: ED's human evaluation; ESConv's human interactive evaluation (Fluency, Identification, Comforting, …); ESC-Eval's "fluency, diversity, empathy, information, humanoid, skillful, overall"; ExTES's Informativeness/Understanding/Helpfulness/Consistency/Coherence with Fleiss' κ.
  - Seeker self-report outcomes: ESConv recorded pre/post negative-emotion intensity and empathy/relevance survey scores.
  - Learned scorers trained on human labels: EPITOME (empathy), PAIR (reflection quality), ESC-RANK.
  - Increasingly, LLM judges, which the owner wants to avoid and which CounselBench found overrate responses.
- **Practical middle ground (my suggestion, not an established standard):**
  - Combine classifier-based proxies: an emotion-acknowledgement rate via a GoEmotions classifier; a strategy-mix distribution via a tagger trained on permissively licensed data; a refusal rate via XSTest string match; and an unsafe-reply rate via a DiaSafety classifier.
  - Add Distinct-n for repetitiveness.
  - Add periodic small-scale blinded human pairwise ratings (e.g. 50–100 conversations, 2–3 raters, report κ) for warmth and naturalness.
