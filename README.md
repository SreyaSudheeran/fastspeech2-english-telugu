# **FastSpeech 2 TTS for English (Svarah) and Telugu (IndicTTS)**

Two independently trained [FastSpeech 2](https://arxiv.org/abs/2006.04558) text-to-speech models, one for **English** and one for **Telugu**, built on [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2). Each language has its own preprocessing, training, inference, configuration and checkpoint, and both use the pretrained universal HiFi-GAN vocoder.

## **Highlights**

|  | English | Telugu |
| :---- | :---- | :---- |
| Dataset | Svarah (Indian-accented English) | IndicTTS Telugu |
| Training audio | 5.00 h (3,457 clips) | 4.69 h (2,566 clips) |
| Validation audio | 1.00 h (698 clips) | 0.94 h (518 clips) |
| Speakers | 115 pseudo-speakers (multi-speaker) | 1 (single-speaker) |
| Alignment | Pretrained MFA `english_us_arpa` | MFA acoustic model trained on the selected data |
| Front-end | POS-aware G2P with a noun/verb heteronym table | Rule-based Telugu G2P (46 phones) with number, time and ordinal expansion |
| Checkpoint used | Step 12,000 (best validation, total loss 2.539) | Step 4,000 (lowest mel and post-net mel loss) |
| Test sentences | 5, all intelligible | 5, clear and natural by ear |

Model size is about 35.4 M parameters. Training ran on a free Google Colab T4 GPU.

## **Project Structure**

```
project/
├── English/
│   ├── preprocess.py
│   ├── train.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/
├── Telugu/
│   ├── preprocess.py
│   ├── train.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/
│   │   └── variance_adaptor_plots/{duration,pitch,energy}/
│   └── Telugu/
│       ├── training_loss.png
│       ├── validation_loss.png
│       ├── test_generated_audios/
│       └── variance_adaptor_plots/{duration,pitch,energy}/
├── Report/
│   └── Technical_Report.pdf
├── common/                  # shared code used by both languages
│   ├── model/  transformer/  audio/  utils/  text/
│   ├── hifigan/  dataset/  evaluate/  preprocessor/
│   ├── trainer.py
│   └── inference_core.py
├── scripts/
│   ├── export_checkpoint.py
│   ├── verify_split.py
│   └── mfa_json_to_textgrid.py
├── test_sentences.json
├── README.md
├── requirements.txt
└── LICENSE_upstream_FastSpeech2
```

`common/` is the only addition to the required layout. Both language folders import the shared upstream-derived modules from it, so the FastSpeech 2 code is not duplicated.

## **Quick Start**

### **1\. Install**

```sh
pip install -r requirements.txt
```

Restart the Python session afterwards so NumPy 1.26.4 is loaded. `requirements.txt` also lists the NLTK data needed.

You will also need:

* **Montreal Forced Aligner 3.4.3**, installed in a separate Conda environment (for example with `condacolab` on Colab).  
* A **Hugging Face access token** for Svarah (accept the dataset terms first).  
* The **universal HiFi-GAN** checkpoint, `common/hifigan/generator_universal.pth.tar.zip` (about 52 MB). It is extracted automatically on first use and is used as is, with no fine-tuning.

### **2\. Run inference with the included checkpoints**

```sh
python English/inference.py
python Telugu/inference.py
```

Both scripts load `final_model.pth.tar` by default. They synthesise the five test sentences for their language and save the audio and the duration, pitch and energy plots under `Results/`.

### **3\. Reproduce a full pipeline**

Shown for English. For Telugu, replace `English` with `Telugu` (see the Telugu notes in section 3).

```sh
python English/preprocess.py --stage prepare    # select, filter, subsample, split, write MFA dictionary
python English/preprocess.py --stage align      # Montreal Forced Aligner
python English/preprocess.py --stage features   # mel, pitch, energy, durations
python scripts/verify_split.py English          # confirm train/validation are disjoint (reads every audio file)
python English/train.py                         # add --resume to continue from the latest checkpoint
python scripts/export_checkpoint.py English/checkpoints/best.pth.tar English/checkpoints/final_model.pth.tar
python English/inference.py
```

If MFA is not installed, the `align` stage prints the exact commands to run.

## **1\. Datasets and Preprocessing**

### **English: Svarah**

Source: [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (about 9.6 h, 117 speakers, read and spontaneous Indian-accented English; only a test split of 6,656 clips is provided).

Svarah has no speaker-ID column, so speaker metadata (gender, age group, native state and district, education, occupation) is hashed into **pseudo-speaker IDs**, giving 115 pseudo-speakers. The model is multi-speaker with a speaker embedding, and inference uses the pseudo-speaker with the most training data.

### **Telugu: IndicTTS**

Source: [SPRINGLab/IndicTTS\_Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) (IIT Madras IndicTTS, Parquet format). The speaker with the most usable audio is selected (`speaker_mode: single_best`) for a clean single-speaker model.

### **Selection and splitting**

| Step | English (Svarah) | Telugu (IndicTTS) |
| :---- | :---- | :---- |
| Transcript filter | Characters `[a-z' ,.?!]` after normalisation; annotation tags removed | Telugu script after normalisation |
| Duration filter | 1 to 15 s | 1 to 15 s |
| Subsampling | Round-robin over speakers | Round-robin |
| Oversampling margin | 1.20 (covers alignment losses) | 1.10 |
| Seed | 1234 | 1234 |
| Final training set | 3,457 utterances, 5.00 h | 2,566 utterances, 4.69 h |
| Final validation set | 698 utterances, 1.00 h | 518 utterances, 0.94 h |

### **Train/validation separation**

* Duplicate transcripts were removed, so no sentence appears in both splits.  
* Training and validation utterance IDs are disjoint, and audio hashes were checked for overlap.  
* Every English validation speaker also appears in the training data.

`scripts/verify_split.py English` and `scripts/verify_split.py Telugu` re-check these guarantees.

### **Audio and feature settings (both languages)**

| Parameter | Value |
| :---- | :---- |
| Sample rate | 22,050 Hz, peak-normalised |
| Mel channels | 80 (0 to 8,000 Hz) |
| Hop length | 256 |
| Pitch / energy | Phoneme-level, z-normalised with **training-set** statistics only |

## **2\. Modifications to the Reference Implementation**

The FastSpeech 2 architecture, Transformer blocks, losses and optimiser are unchanged from the reference. The work went into language support, data preparation and tooling.

| File | Modification |
| :---- | :---- |
| `common/text/symbols.py` | Language-specific symbol tables chosen with `FS2_LANG`; Pinyin removed; Telugu phone inventory added. |
| `common/text/telugu.py` | Telugu NFC normalisation, number/time/ordinal expansion, and deterministic script-to-phone G2P. |
| `common/text/english.py` | POS-aware English G2P with `g2p_en` and a noun/verb heteronym table. |
| `common/text/cleaners.py` | Clock-time and AM/PM expansion, number normalisation, Telugu cleaners. |
| `common/preprocessor/data_prep.py` | Dataset reading, filtering, deduplication, subsampling, splitting and dictionary generation. |
| `common/preprocessor/preprocessor.py` | Explicit split lists, train-only statistics, silence handling, maximum-length filtering, duration-based trimming. |
| `common/audio/*` | librosa 0.10 compatibility; removed hard-coded `.cuda()` calls from the STFT code. |
| `common/utils/model.py` | Module-relative paths, current PyTorch checkpoint loading, automatic vocoder extraction, `--ckpt_file` support. |
| `common/trainer.py` | CSV loss logging, best-validation checkpoint, resume support, loss plots, worker configuration. |
| `common/inference_core.py` | Reusable inference and test-sentence synthesis. |
| `common/utils/plotting.py` | Variance-adaptor plots and CSV exports. |
| `*/config.yaml` | Independent English and Telugu configurations combining preprocessing, model and training settings. |

### **English G2P**

English uses `g2p_en`, the CMU Pronouncing Dictionary and a noun/verb heteronym table. Part-of-speech tags choose the pronunciation, so the two uses of "conduct" in the test set receive different stress: **CON**duct (noun, `K AA1 N D AH0 K T`) and con**DUCT** (verb, `K AH0 N D AH1 K T`). The MFA dictionary contains all CMUdict pronunciation variants.

### **Telugu G2P**

The Telugu front-end handles NFC normalisation, inherent vowels, virama, vowel signs and homorganic anusvara, and expands numbers, times and ordinals into Telugu words:

| Input | Telugu expansion |
| :---- | :---- |
| 2026 | రెండు వేల ఇరవై ఆరు |
| 12వ | పన్నెండవ |
| 10:45 | పది నలభై ఐదు |

The same G2P code writes the MFA dictionary and converts the test sentences, so training and inference phones always agree.

### **Preprocessing improvements**

* MFA silence intervals are kept (`include_empty_intervals=True`) and mapped to `sp`, so pauses are preserved and later phoneme durations stay aligned.  
* Maximum sequence-length filtering and hour-based trimming after alignment.  
* Explicit train/validation lists instead of random shuffling.  
* Fixes for stale-file handling and double normalisation.

## **3\. Telugu Alignment**

No suitable pretrained Telugu MFA acoustic model was assumed, so one was trained on the selected data (3,075 utterances, 46 phones, MFA 3.4.3). The model was backed up after training (`Telugu/mfa_model_backup/`, not included in the repository).

Two MFA behaviours needed small workarounds:

1. MFA's model-card step raised a Jinja template error after the acoustic model had already been saved, so the saved model was copied out directly.  
2. Re-running `mfa align --no_clean` reused the training database and wrote every interval twice, which made the TextGrid export fail on zero-length intervals.

The fix is to export JSON and convert it to TextGrids:

```sh
python Telugu/preprocess.py --stage prepare
mfa train ...                                   # train the Telugu acoustic model
mfa align --output_format json ...
python scripts/mfa_json_to_textgrid.py <json_dir> <TextGrid_dir>
python Telugu/preprocess.py --stage features
python scripts/verify_split.py Telugu
python Telugu/train.py --resume
python scripts/export_checkpoint.py Telugu/checkpoints/best.pth.tar Telugu/checkpoints/final_model.pth.tar
python Telugu/inference.py
```

The conversion script removes exact duplicate intervals, repairs small overlaps and fills gaps with empty-label intervals.

| Conversion result | Value |
| :---- | :---- |
| Files converted | 3,084 |
| Duplicate intervals removed | 309,894 |
| Overlaps repaired | 143 (in 28 files) |

## **4\. Training**

| Parameter | Value |
| :---- | :---- |
| Architecture | FastSpeech 2: 4 encoder \+ 6 decoder FFT blocks, hidden size 256, 2 heads |
| Batch size / optimiser | 16, Adam (0.9, 0.98) |
| Warm-up / gradient clipping | 4,000 steps / 1.0 |
| Validation interval | Every 1,000 steps |
| Vocoder | Universal HiFi-GAN |
| Hardware | Google Colab free T4 GPU |

&nbsp;

| Schedule | English | Telugu |
| :---- | :---- | :---- |
| Configured steps | 60,000 | 10,000 |
| LR drops (factor 0.3) | 40,000 / 50,000 / 55,000 | 7,000 / 8,500 / 9,500 |
| Steps trained | about 25,000 | 4,000 |

Both runs used the free Colab GPU quota and were stopped when it ran out. The repository `config.yaml` files hold the full planned schedules.

**Selected checkpoints.** English uses step 12,000, which had the best validation loss (2.539). Telugu uses step 4,000, which had the lowest mel and post-net mel losses; its best total validation loss (2.232) came at step 2,000.

### **Running on Colab**

* Google Drive stores checkpoints and logs. Reinstall requirements and restart the session after a runtime reset.  
* For fast training, copy `mel/`, `pitch/`, `energy/`, `duration/`, `*.txt` and `*.json` to Colab's local disk and point `preprocess.path.preprocessed_path` at it. Reading thousands of small files from Drive is much slower.  
* TensorBoard is optional. If it cannot be imported, training continues with CSV logs in `<Language>/logs/*.csv`.  
* Feature extraction starts from scratch if it is interrupted.

## **5\. Results**

`Results/` contains, for each language:

* `training_loss.png` and `validation_loss.png` (total loss and the mel, post-net mel, pitch, energy and duration components)  
* `test_generated_audios/`: synthesised audio for all five test sentences  
* `variance_adaptor_plots/`: predicted duration, pitch (F0) and energy for every test sentence, with matching CSV files

| Language | Test sentences | Listening evaluation |
| :---- | :---- | :---- |
| English | 5 (E1, E2a, E2b, E3, E4) | All five fully intelligible, including the rare word, the heteronym pair and the date/time sentence. |
| Telugu | 5 (T1, T2a, T2b, T3, T4) | All five clear and natural by ear. |

The Telugu audio was checked against a Google Translate reference voice. A native-speaker listening test and objective metrics (MCD, F0 RMSE, ASR error rate) are natural next steps.

Detailed analysis of the loss curves, the generated speech and the variance-adaptor plots is in [`Report/Technical_Report.pdf`](https://claude.ai/chat/Report/Technical_Report.pdf).

## **6\. Test Sentences**

The assignment PDF's text layer had extraction problems (broken spacing, displaced Telugu vowel signs), so the sentences were restored from the original document. `test_sentences.json` stores both the corrected text and the raw extraction.

Some test words are outside standard pronunciation dictionaries: English `antidisestablishmentarianism` and `andaroo`, and Telugu `న్యూరోఫార్మకోలాజికల్`. These go through neural G2P (English) or the rule-based G2P (Telugu).

The Telugu word `పట్టు` (grip in T2a, silk in T2b) gets the same phones in both sentences, so any difference in the audio comes from sentence context in the acoustic model.

## **7\. Notes and Possible Extensions**

* **Training length:** both models were trained for part of their planned schedules. Longer training with the configured learning-rate decay is the most direct way to improve them further.  
* **Vocoder:** the universal HiFi-GAN is used unchanged. A Telugu-specific or fine-tuned vocoder could further improve Telugu timbre.  
* **Telugu G2P and alignment:** a larger lexicon and a pretrained multilingual aligner would add coverage for loanwords and sandhi.  
* **Evaluation:** a MOS listening test with native speakers and objective metrics would complement the by-ear checks reported here.

## **8\. Checkpoints**

| Language | File | Training step |
| :---- | :---- | :---- |
| English | `English/checkpoints/final_model.pth.tar` | 12,000 |
| Telugu | `Telugu/checkpoints/final_model.pth.tar` | 4,000 |

Each file holds model weights only (FP16, about 72 MB), which is below GitHub's 100 MB limit, so Git LFS is not needed. Full training checkpoints with optimiser state (about 420 MB each) are not included.

## **9\. Credits and License**

Built on these open-source projects:

* [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2): FastSpeech 2 implementation (MIT)  
* [keithito/tacotron](https://github.com/keithito/tacotron): text-processing components  
* [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2): audio components  
* [jik876/hifi-gan](https://github.com/jik876/hifi-gan): HiFi-GAN vocoder

The upstream license is included as `LICENSE_upstream_FastSpeech2`.

**Reference:** Ren, Y., et al. (2020). *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech.* ICLR. [arXiv:2006.04558](https://arxiv.org/abs/2006.04558)

&nbsp;