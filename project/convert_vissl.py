import torch

# 1. Wpisz tu nazwę pobranego pliku z VISSL
nazwa_pliku_vissl = "models_out/checkpoints/model_final_checkpoint_phase799.torch"

print(f"Wczytywanie modelu VISSL: {nazwa_pliku_vissl}...")
ckpt = torch.load(nazwa_pliku_vissl, map_location="cpu")

# 2. VISSL chowa wagi głęboko w słowniku pod kluczem 'classy_state_dict'
if 'classy_state_dict' in ckpt:
    vissl_state_dict = ckpt['classy_state_dict']['base_model']['model']['trunk']
else:
    vissl_state_dict = ckpt  # Wersja awaryjna, gdyby struktura była płaska

nowe_wagi = {}

# 3. Tłumaczenie nazw warstw
for klucz, wartosc in vissl_state_dict.items():
    # VISSL często ma klucze typu: "_feature_blocks.conv1.weight"
    # solo-learn oczekuje: "backbone.conv1.weight"

    nowy_klucz = klucz.replace('_feature_blocks.', '')

    # Upewniamy się, że klucz zaczyna się od 'backbone.'
    if not nowy_klucz.startswith('backbone.'):
        nowy_klucz = f"backbone.{nowy_klucz}"

    nowe_wagi[nowy_klucz] = wartosc

# 4. Zapisujemy w standardowym formacie solo-learn
sciezka_wyjsciowa = "models_out/checkpoints/simclr-vissl-converted-resnet50-800epochs.pth"
torch.save({"state_dict": nowe_wagi}, sciezka_wyjsciowa)

print(f"Gotowe! Możesz teraz w pliku YAML podać ścieżkę: {sciezka_wyjsciowa}")