import os
import shutil
import urllib.request

# Пути
EMBEDDERS_DIR = os.path.join(os.getcwd(), "rvc", "models", "embedders")
PREDICTORS_DIR = os.path.join(os.getcwd(), "rvc", "models", "predictors")

# URL
BASE_URL_EMB = "https://huggingface.co/Politrees/RVC_resources/resolve/main/embedders/pytorch/"
BASE_URL_PRED = "https://huggingface.co/Politrees/RVC_resources/resolve/main/predictors/"

# Модели: имя и куда ставить
MODELS = {
    "hubert_base.pt": (BASE_URL_EMB, EMBEDDERS_DIR),
    "contentvec_base.pt": (BASE_URL_EMB, EMBEDDERS_DIR),
    "rmvpe.pt": (BASE_URL_PRED, PREDICTORS_DIR),
    # при желании можешь добавить и fcpe.pt сюда:
    # "fcpe.pt": (BASE_URL_PRED, PREDICTORS_DIR),
}


def download_file(url, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with urllib.request.urlopen(url) as response, open(destination, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def progress(value, desc=""):
    print(f"{value*100:.0f}% | {desc}")


def download_and_replace_model(model_name, base_url, target_dir):
    model_url = base_url + model_name
    dst_path = os.path.join(target_dir, model_name)

    progress(0.4, desc=f'[~] Установка модели "{model_name}"...')
    download_file(model_url, dst_path)
    progress(1.0, desc=f'[+] Модель "{model_name}" установлена.')
    return f'Модель "{model_name}" успешно установлена → {dst_path}'


def install_models():
    for model_name, (base_url, target_dir) in MODELS.items():
        result = download_and_replace_model(model_name, base_url, target_dir)
        print(result)


if __name__ == "__main__":
    install_models()
