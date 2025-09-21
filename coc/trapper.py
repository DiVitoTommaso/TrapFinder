import discord
from discord import app_commands
from discord.ext import commands
from datetime import datetime

from models import *
from environment import *

# --------------------
# Load models (multi-type)
# --------------------
def load_models_with_folds(path, points, top_k, num_folds):
    save_path = f"{path}/Top10_hparams.json"
    assert os.path.exists(save_path), f"Top10_hparams.json not found in {path}"

    with open(save_path, "r") as f:
        best_hparams = json.load(f)

    best_hparams = best_hparams[:top_k]
    all_models = []

    for i, hp in enumerate(best_hparams, 1):
        fold_models = []
        for fold in range(1, num_folds + 1):
            model = ResNet(INPUT_SIZE, points, hp["hidden_dim"], hp["num_layers"], 0.0)
            model_file = f"{path}/Model{i}-Fold{fold}.pt"
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Missing model file: {model_file}")
            model.load_state_dict(torch.load(model_file, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            fold_models.append(model)
        all_models.append(fold_models)
    return all_models


# --------------------
# Drawing utilities
# --------------------
def draw_star_pil(image, x, y, size=10, fill_color=(255, 215, 0), outline_color=(0, 0, 0)):
    draw = ImageDraw.Draw(image)
    points = []
    for i in range(10):
        angle = i * np.pi / 5
        r = size if i % 2 == 0 else size / 2
        px = x + r * np.sin(angle)
        py = y - r * np.cos(angle)
        points.append((px, py))
    draw.polygon(points, fill=fill_color, outline=outline_color)
    return image


# --------------------
# Inference utilities
# --------------------
def overlay_stars(image, model, points=1):
    """Run inference with one model and return predicted centers."""
    resize_input = transforms.Resize(INPUT_SIZE[0])
    crop_input = transforms.CenterCrop(INPUT_SIZE)
    to_tensor = transforms.ToTensor()

    image_cropped = crop_input(resize_input(image))
    input_tensor = to_tensor(image_cropped).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)[0].cpu().numpy()  # shape (points, 2)

    preds = []
    for (x, y) in outputs[:points]:
        if x == 0 and y == 0:
            continue
        preds.append((x, y))

    return preds, image_cropped


def overlay_stars_fold_ensemble(image, fold_models, points=1):
    """Ensemble predictions across folds and draw stars."""
    all_preds = []

    for model in fold_models:
        preds, _ = overlay_stars(image, model, points=points)
        if preds:
            all_preds.append(preds)

    if not all_preds:
        return None

    # Average predictions across folds
    all_preds = np.array(all_preds)  # shape (num_folds, num_points, 2)
    avg_preds = np.mean(all_preds, axis=0)  # shape (num_points, 2)

    resize_input = transforms.Resize(INPUT_SIZE[0])
    crop_input = transforms.CenterCrop(INPUT_SIZE)
    image_cropped = crop_input(resize_input(image))

    # Draw stars
    for (x, y) in avg_preds:
        image_cropped = draw_star_pil(image_cropped, x, y, size=10)

    return image_cropped


# --------------------
# Load ensembles
# --------------------
TOP_K = 1
NUM_FOLDS = 4

tornado_models = load_models_with_folds("tornado_models", points=1, top_k=TOP_K, num_folds=NUM_FOLDS)
airbomb_models = load_models_with_folds("airbomb_models", points=7, top_k=TOP_K, num_folds=NUM_FOLDS)

# user usage tracking {user_id: (date, count)}
user_usage = {}


# --------------------
# Discord bot setup
# --------------------
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f"Logged in as {bot.user} and slash commands synced!")


# --------------------
# Generic inference helper
# --------------------
async def run_inference(interaction, image: discord.Attachment, all_models, label_name):
    user_id = interaction.user.id
    today = datetime.utcnow().date()

    if user_id not in user_usage or user_usage[user_id][0] != today:
        user_usage[user_id] = (today, 0)

    if user_usage[user_id][1] >= 5:
        await interaction.response.send_message(
            "❌ You reached the daily limit of 5 inferences.", ephemeral=True
        )
        return

    await interaction.response.defer()
    img_bytes = await image.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    result_img = None
    for fold_models in all_models:
        result_img = overlay_stars_fold_ensemble(pil_img, fold_models)
        if result_img is not None:
            break

    if result_img is None:
        await interaction.followup.send(f"⚠️ No predictions were made for {label_name}.", ephemeral=True)
        return

    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)

    user_usage[user_id] = (today, user_usage[user_id][1] + 1)

    await interaction.followup.send(
        content=f"✨ Prediction result for **{label_name}**",
        file=discord.File(buf, filename=f"{label_name}_result.png")
    )


# --------------------
# Slash Commands
# --------------------
@bot.tree.command(name="locate", description="Run inference on an uploaded image")
@app_commands.describe(image="Attach an image for inference")
async def locate(interaction: discord.Interaction, image: discord.Attachment):
    user_id = interaction.user.id
    today = datetime.utcnow().date()

    # reset usage if new day
    if user_id not in user_usage or user_usage[user_id][0] != today:
        user_usage[user_id] = (today, 0)

    # check quota
    if user_usage[user_id][1] >= 5:
        await interaction.response.send_message(
            "❌ You reached the daily limit of 5 inferences.", ephemeral=True
        )
        return

    # defer response (allows more processing time)
    await interaction.response.defer()

    # read uploaded file
    img_bytes = await image.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    files = []

    # ---- Tornado inference (1 point) ----
    tornado_result = None
    for fold_models in tornado_models:  # ensemble
        tornado_result = overlay_stars_fold_ensemble(pil_img, fold_models, points=1)
        if tornado_result is not None:
            break
    if tornado_result is not None:
        buf1 = io.BytesIO()
        tornado_result.save(buf1, format="PNG")
        buf1.seek(0)
        files.append(discord.File(buf1, filename="tornado_result.png"))

    # ---- Airbomb inference (7 points) ----
    airbomb_result = None
    for fold_models in airbomb_models:  # ensemble
        airbomb_result = overlay_stars_fold_ensemble(pil_img, fold_models, points=7)
        if airbomb_result is not None:
            break
    if airbomb_result is not None:
        buf2 = io.BytesIO()
        airbomb_result.save(buf2, format="PNG")
        buf2.seek(0)
        files.append(discord.File(buf2, filename="airbomb_result.png"))

    if not files:
        await interaction.followup.send("⚠️ No predictions were made.", ephemeral=True)
        return

    # update usage
    user_usage[user_id] = (today, user_usage[user_id][1] + 1)

    # send both images in one reply
    await interaction.followup.send(
        content="✨ Prediction results (Tornado + Airbomb):",
        files=files
    )

# --------------------
# Run bot
# --------------------
if __name__ == "__main__":
    bot.run('TOKEN')