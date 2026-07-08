# ============================================================
# DENEME 6: Dropout + LR Scheduler
# Kocaeli Üniversitesi - Lisans Bitirme Projesi
# Öğrenci: Zülal KAYA
# ============================================================
#
# ──────────────────────────────────────────────────────────
# PROJE GEÇMİŞİ VE GELİŞİM SÜRECİ
# ──────────────────────────────────────────────────────────
#
# DENEME 1 - Çerçeve Tabanlı Bölme (YANLIŞ YAKLAŞIM)
# ─────────────────────────────────────────────────────
# Ne yapıldı:
#   - Her videodan 10 çerçeve çıkarıldı
#   - Tüm çerçeveler RASTGELE eğitim/val/test kümelerine ayrıldı
#   - EfficientNet-B0, pretrained=True
# Sonuç: %99 doğruluk, AUC 0.9999
# Sorun: VERİ SIZINTISI!
#   → Aynı videonun farklı çerçeveleri hem train hem test'te!
#   → Model test'te eğitimde gördüğü çerçeveleri tanıdı
#   → %99 gerçek başarım değil, kandırmaca!
# Ders: Çerçeve tabanlı bölme ASLA kullanılmamalı!
#
# DENEME 2 - Video Tabanlı Bölme (DOĞRU YAKLAŞIM)
# ─────────────────────────────────────────────────
# Ne yapıldı:
#   - Önce VİDEOLAR kümelere ayrıldı (%70/%15/%15)
#   - Bir videonun tüm çerçeveleri sadece bir kümede
#   - Aynı kaynak videonun real ve fake'i aynı kümede
# Sonuç: %68 doğruluk, AUC 0.7377
# Sorun: Düşük başarım, overfitting başlıyor
# Ders: Gerçekçi değerlendirme için video tabanlı bölme şart!
#       %99→%68 düşüşü veri sızıntısının ne kadar yanıltıcı
#       olduğunu açıkça gösterdi.
#
# DENEME 3 - Veri Artırma + 20 Çerçeve
# ─────────────────────────────────────
# Ne yapıldı:
#   - Çerçeve sayısı 10→20'ye çıkarıldı
#   - Sadece eğitime augmentation eklendi:
#     * RandomHorizontalFlip (p=0.5)
#     * RandomRotation(10 derece)
#     * ColorJitter (brightness=0.2, contrast=0.2)
# Sonuç: %83 doğruluk, AUC 0.8858
# Sorun: Overfitting devam ediyor
#   → Train loss: 0.03, Val loss: 0.44-1.20 (çok büyük fark!)
# Ders: Augmentation yardımcı oldu ama overfitting tam çözülmedi
#
# DENEME 4 - Tüm Çerçeveler (~366K)
# ─────────────────────────────────────
# Ne yapıldı:
#   - Her videodan TÜM çerçeveler çıkarıldı
#   - Video başına ortalama 450 çerçeve
#   - Toplam ~366.000 çerçeve
# Sonuç: %88 doğruluk, AUC 0.9321
# Sorun: Overfitting hâlâ var, eğitim çok uzun sürüyor
#   → Train loss: 0.01, Val loss: 0.32-0.87
# Ders: Daha fazla veri yardımcı oldu ama yeterli değil.
#       366K çerçeve Google Colab'da saatlerce sürdü.
#       Çerçeveler arasındaki yüksek görsel benzerlik
#       overfitting'i tam çözmüyor.
#
# DENEME 5 - MTCNN Yüz Kırpma + 50 Çerçeve (EN İYİ)
# ────────────────────────────────────────────────────
# Ne yapıldı:
#   - MTCNN ile otomatik yüz kırpma eklendi
#   - Video başına EŞİT ARALIKLI 50 çerçeve
#   - Yüz bulunamazsa çerçeve atlandı
#   - Her kaynak videonun sadece 1 fake karşılığı kullanıldı
#     (328 gerçek-sahte çift)
# Sonuç: %97 doğruluk, AUC 0.9969 ← BÜYÜK SIÇRAMA!
# Sorun: Overfitting hâlâ var!
#   → Val loss dalgalı, train-val farkı büyük
#   → Val loss düşük olmamasına rağmen test %97 → ŞÜPHELI!
#     (Test kümesi küçük, şanslı bölünme olabilir)
# Ders: Yüz kırpma en etkili iyileştirme oldu!
#   Deepfake manipülasyonu yüz bölgesinde olduğu için
#   model arka plandan değil yüzden öğreniyor.
#   Ancak overfitting sorunu çözülmedi → Deneme 6'ya geçildi.
#
# ──────────────────────────────────────────────────────────
# DENEME 6 - Bu Dosya
# ──────────────────────────────────────────────────────────
# Amaç: Deneme 5'teki overfitting sorununu gidermek
#
# Yapılan değişiklikler:
#   1. Dropout %40 eklendi (modelin son katmanına)
#      → Nöronların %40'ı eğitimde rastgele kapatılır
#      → Model ezberleme yerine genelleme öğrenir
#
#   2. ReduceLROnPlateau öğrenme oranı zamanlayıcısı eklendi
#      → Val loss 3 epoch iyileşmezse lr 0.5x azalır
#      → Model "yerinde saymasını" engeller
#
#   3. Doğrulama kümesi büyütüldü
#      → %70/%15/%15 → %55/%30/%15
#      → Daha büyük val kümesi → daha güvenilir overfitting tespiti
#
#   4. Augmentation kaldırıldı
#      → Dropout tek başına yeterli mi test edildi
#
#   5. Frame tekrar kontrolü eklendi
#      → Colab kopsa bile frame'leri yeniden çıkarmaz
#
# Sonuç: %97.96 doğruluk, AUC 0.9980
#   → En önemli gelişme: train-val loss farkı çok azaldı!
#   → Epoch 9: Train loss 0.0134, Val loss 0.0210 (çok yakın!)
#   → Val accuracy %99 → önceki deneylere göre çok daha stabil
#
# Karşılaşılan hatalar ve çözümler:
#   ❌ Pillow ImportError → !pip install Pillow==10.2.0 ile çözüldü
#   ❌ num_workers=2 Colab'da kilitlenme → num_workers=0 yapıldı
#   ❌ face.permute().numpy() GPU hatası → .cpu() eklendi
#   ❌ verbose=True DeprecationWarning → kaldırıldı
# ============================================================

# ──────────────────────────────────────────────────────────
# HÜCRE 1 - Drive Bağla
# ──────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ──────────────────────────────────────────────────────────
# HÜCRE 2 - Kütüphaneler
# ──────────────────────────────────────────────────────────
!pip install Pillow==10.2.0 -q          # Önemli: önce Pillow!
!pip install timm facenet-pytorch scikit-learn -q

import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import timm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from facenet_pytorch import MTCNN
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

# ──────────────────────────────────────────────────────────
# HÜCRE 3 - Yollar
# ──────────────────────────────────────────────────────────
BASE_PATH   = '/content/drive/MyDrive/deepfake_project/ff_data'
FACE_PATH   = '/content/drive/MyDrive/deepfake_project/frames_deney6'
MODEL_PATH  = '/content/drive/MyDrive/deepfake_project/models/deney6'
RESULT_PATH = '/content/drive/MyDrive/deepfake_project/results/deney6'

os.makedirs(os.path.join(FACE_PATH, 'real'), exist_ok=True)
os.makedirs(os.path.join(FACE_PATH, 'fake'), exist_ok=True)
os.makedirs(MODEL_PATH,  exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)

REAL_VIDEO = os.path.join(BASE_PATH,
    'original_sequences', 'youtube', 'c23', 'videos')
FAKE_VIDEO = os.path.join(BASE_PATH,
    'manipulated_sequences', 'Deepfakes', 'c23', 'videos')

# ──────────────────────────────────────────────────────────
# HÜCRE 4 - Video Tabanlı Bölme (SIZINTISIZ)
# Deneme 2'den bu yana kullandığımız doğru yaklaşım
# ──────────────────────────────────────────────────────────
random.seed(42)

real_videos = sorted(os.listdir(REAL_VIDEO))
fake_videos = sorted(os.listdir(FAKE_VIDEO))

# Her kaynak video sadece BİR kez kullanılır
# → Birden fazla fake'i olan videoların sadece ilk fake'i alınır
seen_reals = set()
pairs = []
for fv in fake_videos:
    src = fv.split('_')[0] + '.mp4'
    if src in real_videos and src not in seen_reals:
        pairs.append((src, fv))
        seen_reals.add(src)

print(f"Toplam çift: {len(pairs)}")
random.shuffle(pairs)
n       = len(pairs)
n_train = int(n * 0.55)   # Deneme 5'te %70'ti, büyüttük
n_val   = int(n * 0.30)   # Deneme 5'te %15'ti, büyüttük

train_pairs = pairs[:n_train]
val_pairs   = pairs[n_train : n_train + n_val]
test_pairs  = pairs[n_train + n_val:]

# Sızıntı kontrolü - assert ile otomatik doğrulama
tr = set(p[0] for p in train_pairs)
vr = set(p[0] for p in val_pairs)
te = set(p[0] for p in test_pairs)
assert len(tr & vr) == 0, "SIZINTI: train-val!"
assert len(tr & te) == 0, "SIZINTI: train-test!"
assert len(vr & te) == 0, "SIZINTI: val-test!"
print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")
print("Sızıntı kontrolü geçti! ✅")

# ──────────────────────────────────────────────────────────
# HÜCRE 5 - MTCNN Başlat
# ──────────────────────────────────────────────────────────
mtcnn = MTCNN(
    image_size=224,
    margin=20,        # Yüz sınırına 20px boşluk → sınır bölgesi dahil
    keep_all=False,   # Birden fazla yüzde sadece en büyüğü al
    device=device
)
print("MTCNN hazır!")

# ──────────────────────────────────────────────────────────
# HÜCRE 6 - Frame Çıkarma
# Deneme 5'e ek olarak: frame tekrar kontrolü eklendi
# Colab kopsa bile yeniden başlamak zorunda kalma!
# ──────────────────────────────────────────────────────────
def frames_already_exist(video_id, split_name, label_dir):
    """Bu videonun frame'leri zaten çıkarıldı mı?"""
    folder = os.path.join(FACE_PATH, label_dir)
    if not os.path.exists(folder):
        return False
    existing = [f for f in os.listdir(folder)
                if f.startswith(f"{split_name}_{video_id}__")]
    return len(existing) > 0

def extract_faces(video_path, output_dir, split_name, video_id, n_frames=50):
    # Zaten varsa atla
    label_dir = 'real' if 'real' in output_dir else 'fake'
    if frames_already_exist(video_id, split_name, label_dir):
        return 0

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 10:
        cap.release()
        return 0

    # Baştan 25, ortadan 12, sondan 13 frame
    n_s = n_frames // 2
    n_m = n_frames // 4
    n_e = n_frames - n_s - n_m
    s = np.linspace(0,          total//3,     n_s, dtype=int)
    m = np.linspace(total//3,   2*total//3,   n_m, dtype=int)
    e = np.linspace(2*total//3, total-1,      n_e, dtype=int)
    indices = np.unique(np.concatenate([s, m, e]))

    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face      = mtcnn(Image.fromarray(frame_rgb))
        if face is not None:
            # .cpu() önemli → GPU bellek hatasını önler
            face_np = ((face.permute(1,2,0).cpu().numpy()+1)/2*255
                       ).clip(0,255).astype(np.uint8)
            fname = f"{split_name}_{video_id}__{idx:05d}.jpg"
            cv2.imwrite(os.path.join(output_dir, fname),
                        cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR))
            saved += 1
    cap.release()
    return saved

def process_split(pair_list, split_name):
    print(f"\n{split_name} işleniyor ({len(pair_list)} çift)...")
    for i, (real_v, fake_v) in enumerate(pair_list):
        if i % 20 == 0:
            print(f"  {i}/{len(pair_list)}...")
        extract_faces(
            os.path.join(REAL_VIDEO, real_v),
            os.path.join(FACE_PATH, 'real'),
            split_name, real_v[:-4])
        extract_faces(
            os.path.join(FAKE_VIDEO, fake_v),
            os.path.join(FACE_PATH, 'fake'),
            split_name, fake_v[:-4])
    print(f"{split_name} tamamlandı!")

process_split(train_pairs, 'train')
process_split(val_pairs,   'val')
process_split(test_pairs,  'test')

print(f"\nReal: {len(os.listdir(os.path.join(FACE_PATH,'real')))}")
print(f"Fake: {len(os.listdir(os.path.join(FACE_PATH,'fake')))}")

# ──────────────────────────────────────────────────────────
# HÜCRE 7 - Dataset ve DataLoader
# Augmentation YOK (Deneme 5'ten farklı)
# num_workers=0 (Colab'da 2 kilitlenmeye yol açıyordu)
# ──────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class DeepfakeDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def get_files(folder, label, prefix):
    return [(os.path.join(folder, f), label)
            for f in os.listdir(folder)
            if f.startswith(prefix + '_')]

real_dir = os.path.join(FACE_PATH, 'real')
fake_dir = os.path.join(FACE_PATH, 'fake')

train_files = (get_files(real_dir, 0, 'train') + get_files(fake_dir, 1, 'train'))
val_files   = (get_files(real_dir, 0, 'val')   + get_files(fake_dir, 1, 'val'))
test_files  = (get_files(real_dir, 0, 'test')  + get_files(fake_dir, 1, 'test'))

print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

# num_workers=0 → Colab'da kararlı çalışır
train_loader = DataLoader(DeepfakeDataset(train_files, transform),
                          batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(DeepfakeDataset(val_files,   transform),
                          batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(DeepfakeDataset(test_files,  transform),
                          batch_size=32, shuffle=False, num_workers=0)

# ──────────────────────────────────────────────────────────
# HÜCRE 8 - Model (EfficientNet-B0 + Dropout %40)
# YENİ: Dropout eklendi → overfitting azaltmak için
# ──────────────────────────────────────────────────────────
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)

# YENİ: Dropout %40 + Xavier init
linear = nn.Linear(1280, 2)
nn.init.xavier_uniform_(linear.weight)
nn.init.zeros_(linear.bias)

model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),   # Eğitimde %40 nöron rastgele kapatılır
    linear
)
model = model.to(device)
print(f"Model hazır! Parametre: {sum(p.numel() for p in model.parameters()):,}")

# ──────────────────────────────────────────────────────────
# HÜCRE 9 - Eğitim
# YENİ: ReduceLROnPlateau zamanlayıcısı
# → Val loss 3 epoch iyileşmezse lr 0.5x azalır
# ──────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# YENİ: LR Scheduler
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=3,    # 3 epoch bekle
    factor=0.5,    # lr'yi yarıya indir
    # verbose=True → DeprecationWarning veriyor, kaldırıldı
)

EPOCHS, PATIENCE = 30, 7
best_loss, wait  = float('inf'), 0
t_losses, v_losses, t_accs, v_accs = [], [], [], []

for epoch in range(EPOCHS):
    # Train
    model.train()
    tl, correct, total = 0, 0, 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        tl += loss.item()
        _, pred = torch.max(out, 1)
        correct += (pred == lbls).sum().item()
        total   += lbls.size(0)
    avg_t = tl / len(train_loader)
    acc_t = 100 * correct / total

    # Val
    model.eval()
    vl, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out  = model(imgs)
            loss = criterion(out, lbls)
            vl  += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == lbls).sum().item()
            total   += lbls.size(0)
    avg_v = vl / len(val_loader)
    acc_v = 100 * correct / total

    t_losses.append(avg_t); v_losses.append(avg_v)
    t_accs.append(acc_t);   v_accs.append(acc_v)
    scheduler.step(avg_v)
    lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train {avg_t:.4f}/{acc_t:.1f}% | "
          f"Val {avg_v:.4f}/{acc_v:.1f}% | LR {lr:.6f}")

    if avg_v < best_loss:
        best_loss = avg_v; wait = 0
        torch.save(model.state_dict(),
                   os.path.join(MODEL_PATH, 'best_deney6.pth'))
        print(f"  → Kaydedildi! Val Loss: {best_loss:.4f}")
    else:
        wait += 1
        print(f"  → İyileşme yok ({wait}/{PATIENCE})")
        if wait >= PATIENCE:
            print("Erken durdurma!")
            break

print("Eğitim tamamlandı!")

# ──────────────────────────────────────────────────────────
# HÜCRE 10 - Test
# ──────────────────────────────────────────────────────────
model.load_state_dict(torch.load(
    os.path.join(MODEL_PATH, 'best_deney6.pth')))
model.eval()

all_lbl, all_pred, all_prob = [], [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        out   = model(imgs)
        probs = torch.softmax(out, dim=1)[:, 1]
        _, pred = torch.max(out, 1)
        all_lbl.extend(lbls.cpu().numpy())
        all_pred.extend(pred.cpu().numpy())
        all_prob.extend(probs.cpu().numpy())

auc = roc_auc_score(all_lbl, all_prob)
acc = 100 * sum(p==l for p,l in zip(all_pred,all_lbl)) / len(all_lbl)
cm  = confusion_matrix(all_lbl, all_pred)

print(f"\n{'='*50}")
print(f"TEST SONUÇLARI - DENEME 6")
print(f"{'='*50}")
print(f"Doğruluk : %{acc:.2f}")
print(f"AUC      : {auc:.4f}")
print(classification_report(all_lbl, all_pred,
                             target_names=['Gerçek', 'Sahte']))
print(f"Karışıklık Matrisi:\n{cm}")

# ──────────────────────────────────────────────────────────
# HÜCRE 11 - Grafik
# ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Deneme 6 - Dropout + LR Scheduler\n'
             f'Acc:%{acc:.2f} | AUC:{auc:.4f}',
             fontsize=14, fontweight='bold')

axes[0].plot(t_losses, 'b-o', markersize=3, label='Train Loss')
axes[0].plot(v_losses, 'r-o', markersize=3, label='Val Loss')
axes[0].set_title('Loss Eğrileri')
axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(True)

axes[1].plot(t_accs, 'b-o', markersize=3, label='Train Acc')
axes[1].plot(v_accs, 'r-o', markersize=3, label='Val Acc')
axes[1].set_title('Accuracy Eğrileri')
axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(True)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Gerçek', 'Sahte']
).plot(ax=axes[2], cmap='Blues', colorbar=False)
axes[2].set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, 'results_deney6.png'), dpi=150)
plt.show()
print("Grafik kaydedildi!")
