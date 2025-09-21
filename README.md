# Clash Of Clans – Trap Finder

A **Discord Bot** that detects hidden traps in Clash of Clans villages from a screenshot.

---

## 🚀 How it Works

The bot uses **AI models** trained on a private dataset. To use it:

1. Attach a screenshot of the base you want to analyze.
2. Run the bot’s `/locate` command.
3. The bot will predict and mark the hidden traps on the image.

---

## 📖 Usage Policy

* The project is **open source**.
* You can **host the bot for free** on your own server.

---

## ⚙️ Setup

1. **Create a Discord bot** in the [Discord Developer Portal](https://discord.com/developers/applications).
2. **Create a Conda environment** to avoid breaking other projects:

   ```bash
   conda create -n coc python=3.10
   conda activate coc
   ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
4. **Train the model**:

> ⚠️ Model files are **not included** and the dataset is **private**.
> You will need to create your own dataset. We provide a small portion of our tested dataset as an example to show the expected folder structure and JSONL label format.

* Labels must be in a **JSONL file** with the correct format.

---

## 📁 Expected Dataset Structure

```text
Data/
├── train/
│   ├── img1.png
│   ├── img2.png
│   └── ...
├── val/
│   ├── img1.png
│   └── ...
labels/
├── train.jsonl
├── val.jsonl
```

Each JSONL line corresponds to an image and contains the coordinates of hidden traps.

---

## 💡 Notes

* If you don’t want to spend time on building a dataset, preprocessing it, or handling the training loop, you can **contact me**.
* We can discuss a monthly subscription price to:

  * Create the bot for you and handle training process
  * Improve it periodically for each new game update
  * Maintain the server required to host the bot
