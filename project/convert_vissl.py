import torch


nazwa_pliku_vissl = "models_out/checkpoints/model_final_checkpoint_phase799.torch"

print(f"Wczytywanie modelu VISSL: {nazwa_pliku_vissl}...")
ckpt = torch.load(nazwa_pliku_vissl, map_location="cpu")

# look for key : 'classy_state_dict' in vssl
if 'classy_state_dict' in ckpt:
    vissl_state_dict = ckpt['classy_state_dict']['base_model']['model']['trunk']
else:
    vissl_state_dict = ckpt

nowe_wagi = {}

# translate layers names
for klucz, wartosc in vissl_state_dict.items():
   # keys --> "_feature_blocks.conv1.weight"
    # solo-learn keys -->  "backbone.conv1.weight"

    nowy_klucz = klucz.replace('_feature_blocks.', '')

    # key starts with --> 'backbone.'
    if not nowy_klucz.startswith('backbone.'):
        nowy_klucz = f"backbone.{nowy_klucz}"

    nowe_wagi[nowy_klucz] = wartosc

# save in solo-learn format
sciezka_wyjsciowa = "models_out/checkpoints/simclr-vissl-converted-resnet50-800epochs.pth"
torch.save({"state_dict": nowe_wagi}, sciezka_wyjsciowa)

print(f"Gotowe! Możesz teraz w pliku YAML podać ścieżkę: {sciezka_wyjsciowa}")