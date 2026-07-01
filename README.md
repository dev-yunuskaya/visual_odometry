# İkinci Görev (Pozisyon Tespiti) — Sıfırdan Yol Haritası

> **Varsayım:** Python biliyorsun, temel programlama tecrüben var ama bilgisayarlı görü / SLAM konusunda yenisin. Yol haritası Python + OpenCV (+ istersen ileride PyTorch) üzerine kurulu. Farklıysa söyle, uyarlarım.

---

## 0. Önce görevi doğru sökelim (bu adımı atlama)

Şartnameden (2.2 ve Tablo 6) çıkan gerçek problem şu:

- Kameran neredeyse dik (70-90°) aşağı bakıyor, sen aracın **ilk kareye göre x,y,z yer değiştirmesini** (metre) tahmin ediyorsun.
- Her karede sana `health_status` (sağlık) bilgisi geliyor:
  - **health=1**: ilk 1 dakika (450 kare) garanti sağlıklı — yani sana **gerçek referans değeri** veriliyor. Bu aralıkta istersen kendi tahminini, istersen sunucudan gelen değeri gönderebilirsin.
  - **health=0**: son 4 dakikada (1800 kare) ne zaman başlayıp ne kadar süreceği belirsiz şekilde "GPS kaybı" simüle ediliyor — burada **zorunlu olarak kendi algoritman** konuşacak.
- Puan, tahminin ile gerçek pozisyon arasındaki **ortalama Öklid mesafesi hatası** (Denklem 2) üzerinden hesaplanıyor.

**Buradan çıkan en kritik strateji (işin mutfağı tam burası):**
Saf monoküler VO'nun en büyük laneti **ölçek belirsizliğidir** (tek kameradan piksel hareketini kaç metreye karşılık geldiğini bilemezsin). Ama bu yarışma sana bunu bedavaya çözüyor: health=1 pencerelerinde gerçek metre cinsinden yer değiştirmeyi zaten biliyorsun. Yani asıl işin "boşlukta pozisyon kestirmek" değil, **health=1 anlarını kalibrasyon sinyali olarak kullanıp health=0 boşluklarını doldurmak.** Bu, klasik "GPS kesintisinde görsel odometri ile devam etme" (dead-reckoning) problemidir — robotik literatüründe çok çalışılmış, iyi bilinen bir konu. Bunu bilmek, tekerleği yeniden icat etmeni engeller.

---

## Faz Özeti (öncelik sırası)

| Faz | Konu | Neden şart | Yaklaşık süre |
|---|---|---|---|
| 1 | Kamera geometrisi temelleri | Her yöntemin altyapısı, kamera parametreleri sana verilecek | 3-5 gün |
| 2 | Klasik CV baseline (OpenCV) | Hızlı çalışan ilk sonuç, hata kaynaklarını görmek için | 1 hafta |
| 3 | Aşağı-bakan kamera özel stratejisi | Bu görevin gerçek çekirdeği (ölçek + irtifa) | 1-2 hafta |
| 4 | Durum kestirimi / filtreleme | Drift kontrolü, health geçişlerini yönetmek | 1 hafta |
| 5 | Derin öğrenme (sağlamlaştırma) | Bulanıklık, kar/yağmur, termal kamera gibi bozulmalara dayanıklılık | 2-3 hafta |
| 6 | Veri & simülasyon | Resmi veri seti yayınlanana kadar test/eğitim verisi | Paralel yürür |
| 7 | Kendi değerlendirme sistemin | Kör optimizasyon yapmamak için | 2-3 gün (erken kur) |
| 8 | İterasyon | Sürekli | Yarışmaya kadar |

---

## Faz 1 — Matematiksel / Geometrik Temel

**Neden:** Kamera parametreleri (intrinsic) sana paylaşılacak (2.2.1). Bu matrisi kullanamazsan hiçbir VO yöntemi doğru çalışmaz. Pinhole kamera modeli, homojen koordinatlar, dönüş matrisleri, homografi/esansiyel matris kavramlarını bilmeden ne klasik ne de derin öğrenme yaklaşımı anlamlı olur.

**Öğrenmen gerekenler:**
- Pinhole kamera modeli, intrinsic (K) ve extrinsic (R,t) parametreler
- Homojen koordinatlar, projeksiyon
- Homografi (H) ve esansiyel matris (E) — ne zaman hangisi kullanılır
- Temel epipolar geometri

**Kaynaklar:**
- *Multiple View Geometry in Computer Vision* — Hartley & Zisserman (ilk 6-9 bölüm yeter, "İncil" kitap ama ağır; sadece gerekli kısımları oku)
- Cyrill Stachniss (Bonn Üniversitesi) — YouTube'daki "Photogrammetry & Robotics" ve fotogrametri/SLAM ders serisi (Türkçe altyazı yok ama çok net anlatım, ücretsiz)
- OpenCV resmi dokümantasyonu → "Camera Calibration and 3D Reconstruction" bölümü (docs.opencv.org)
- Coursera: Univ. of Pennsylvania'nın Robotics Specialization serisindeki "Robotics: Perception" dersi (ücretsiz izlenebilir, sertifika ücretli)

---

## Faz 2 — Klasik Bilgisayarlı Görü ile Baseline (Hızlı Prototip)

**Neden:** Şartname videolarda bulanıklık, ölü piksel, kare donması/kaybı olabileceğini açıkça söylüyor (2.1). Karmaşık bir modele atlamadan önce basit, açıklanabilir bir baseline kurup nerede kırıldığını görmek, sonraki fazlarda neye öncelik vereceğini belirler. Ayrıca bu, "mutfaktan başlamak" istediğin kısım — elini koda değdirmenin en hızlı yolu.

**Yapman gerekenler:**
- ORB / SIFT ile özellik (feature) tespiti ve eşleştirme
- Optik akış: Lucas-Kanade (seyrek) ve Farneback (yoğun)
- İki kare arası hareketi essential matrix / homography ile çözüp `cv2.recoverPose` ile R,t çıkarma
- Basit bir "frame-to-frame" VO döngüsü yazıp konumu biriktirerek (integrate ederek) yörünge çizdirme

**Kaynaklar:**
- OpenCV-Python resmi tutorial'ları: "Feature Detection and Description", "Optical Flow" bölümleri
- Scaramuzza & Fraundorfer, *"Visual Odometry" Part I & Part II* (IEEE Robotics & Automation Magazine, 2011-2012) — alanın klasik, kısa ve öz tutorial makaleleri; VO'nun "nasıl çalışır"ını en iyi özetleyen kaynak
- GitHub'da minimal monoküler VO örnek projeleri (arama terimi: "monocular visual odometry OpenCV python") — kodu satır satır anlayarak incele, kopyala-yapıştır yapma

---

## Faz 3 — Aşağı Bakan Kamera İçin Özel Strateji (Asıl Çekirdek)

**Neden:** Standart VO literatürünün çoğu ileri bakan (araba, insan gözü perspektifi) kameralar için yazılmış. Senin kameran neredeyse yere dik bakıyor ve sahne genelde yerel olarak düzlemsel (yol, tarla, çatı). Bu, problemi **basitleştirir** — homografi tabanlı yaklaşımlar (drone optik akış sensörlerinin, örn. PX4Flow'un, mantığına çok yakın) essential matrix'ten daha stabil çalışır.

**Çözülmesi gereken iki alt problem:**

1. **x, y (yatay hareket):** Ardışık kareler arası homografiyi çıkar, düzlemsel sahne varsayımıyla dönüşü ayıkla, kalan öteleme bileşenini piksel cinsinden al.
2. **z (irtifa değişimi) + ölçek:** Tek kamerada en zor kısım budur. İki pratik yol:
   - Eşleşen özellik noktaları arasındaki ortalama mesafenin kareler arası **büyüme/küçülme oranı**, irtifa değişimiyle orantılıdır (yaklaşırsan nesneler büyür). Bu "optic flow divergence / focus of expansion" mantığı, arıların irtifa algısı gibi biyoloji-esinli UAV kontrolünde de kullanılır.
   - **UAP/UAİ alanları 4,5 metre çapında sabit boyutlu** (2.1.2) — görüntüde bu daireleri gördüğünde gerçek ölçek için harika bir referans nesnesi olarak kullanabilirsin.

3. **En önemlisi — piksel→metre kalibrasyonu:** health=1 penceresinde (ilk 450 kare) hem görüntüyü hem gerçek metre cinsinden yer değiştirmeyi biliyorsun. Bu pencereyi kullanarak "piksel hareketi → gerçek metre" dönüşümünü (session'a, yani o anki irtifa/kamera ayarına özel) **regresyon ile kalibre et**. Bu, saf monoküler ölçek belirsizliğini yarı-gözetimli bir probleme çeviriyor — yarışmanın tasarımındaki en büyük avantaj bu.

**Kaynaklar:**
- PX4Flow projesi dokümantasyonu (aşağı bakan optik akış sensörünün prensip mantığı — donanım değil ama algoritma mantığı doğrudan işine yarar)
- Arama terimleri: *"optical flow based altitude estimation UAV"*, *"homography based visual odometry downward facing camera"*, *"time to contact optical flow"*
- "Vision-based state estimation for autonomous rotorcraft MAVs" başlıklı akademik makaleler (Google Scholar'da bu terimle arama yap, konuya en yakın literatür burada)

---

## Faz 4 — Durum Kestirimi / Filtreleme (Drift Kontrolü)

**Neden:** Kare-kare hareket tahminleri gürültülüdür; bunları düz toplarsan hata katlanarak büyür (drift). Puanlama tüm kareler üzerinden **ortalama** hataya bakıyor, yani uzun health=0 bloklarında drift kontrolsüz büyürse skorun çöker. Ayrıca health bayrağı 0↔1 arasında geçiş yapabileceği için (2.2.2), bu geçişleri yumuşak yönetecek bir filtre mimarisine ihtiyacın var.

**Yapman gerekenler:**
- Basit bir Kalman Filtresi (veya complementary filter) ile ardışık VO tahminlerini birleştirip pürüzsüz bir yörünge üret
- health=1 anlarında filtreyi gerçek değere "resetle/düzelt", health=0'da sadece kendi tahminlerinle devam et (dead-reckoning)
- Aşırı güvenilmeyen (düşük eşleşme sayısı, bulanık kare gibi) tahminleri filtrenin ölçüm gürültüsü olarak ağırlıklandır

**Kaynaklar:**
- *Kalman and Bayesian Filters in Python* — Roger R. Labbe Jr. (GitHub'da ücretsiz, interaktif Jupyter kitap — bu konuda en pratik, en az matematik korkutucu kaynak)
- *Probabilistic Robotics* — Thrun, Burgard, Fox (daha akademik, Kalman/Particle filter'ın robotik bağlamdaki standart referansı)
- Cyrill Stachniss'in aynı YouTube kanalındaki state estimation / Kalman filter dersleri

---

## Faz 5 — Derin Öğrenme ile Sağlamlaştırma (İleri Seviye)

**Neden:** Şartname açıkça diyor ki görüntüler: kar/yağmur olabilir, şehir/orman/deniz üzerinde çekilebilir, RGB veya termal olabilir, bulanıklık/ölü piksel/kare donması içerebilir (2.1). Klasik özellik eşleştirme (ORB/SIFT) bu koşullarda kolayca kırılır — özellikle termalde veya düşük kontrastlı sahnelerde (deniz üstü!) eşleşecek "köşe" bulamayabilirsin. Derin öğrenme tabanlı akış/VO modelleri bu tür bozulmalara karşı daha dayanıklı olabilir. Şartname "algoritma hızı puanlanmıyor, saniyede 1 kare işlesen yeterli" diyor (Bölüm 7) — bu da ağır modelleri kullanabileceğin anlamına geliyor.

**Seçeneklerin (kolaydan zora):**
1. Klasik geometriyi koru, sadece özellik eşleştirmeyi güçlü bir derin optik akış ağıyla değiştir (hibrit yaklaşım)
2. Uçtan uca öğrenen VO modeli (kare çifti → doğrudan öteleme tahmini)
3. Kendi kendini denetleyen (self-supervised) derinlik+poz ağları — etiketli hava aracı veri seti azlığı sorununu çözer çünkü etiketsiz videoyla eğitilebilir

**Kaynaklar:**
- RAFT — *"Recurrent All-Pairs Field Transforms for Optical Flow"* (Teed & Deng, 2020) — güncel, güçlü optik akış ağı, GitHub: `princeton-vl/RAFT`
- DeepVO — *"Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks"* (Wang ve ark., 2017)
- TartanVO — CMU'nun genelleştirilebilir öğrenen VO modeli (Wang ve ark., 2020), TartanAir veri setiyle birlikte yayınlandı
- Monodepth2 — *"Digging Into Self-Supervised Monocular Depth Estimation"* (Godard ve ark., 2019), GitHub: `nianticlabs/monodepth2` — self-supervised derinlik+poz mimarisini anlamak için iyi bir başlangıç

---

## Faz 6 — Veri ve Simülasyon

**Neden:** Resmi örnek veri seti GitHub üzerinden paylaşılacak (Bölüm 10) ama muhtemelen sınırlı olacak. Model geliştirme/test için ek veriye ihtiyacın olacak, özellikle derin öğrenmeye gidersen.

**Kaynaklar:**
- **Microsoft AirSim** — Unreal Engine tabanlı simülatör; aşağı bakan kamera + gerçek zemin doğrusu (ground truth) poz üretebiliyor, tam da bu görevin ihtiyacı olan senaryoyu sentetik olarak kurabilirsin
- **TartanAir** (CMU) — zorlu ortamlar için sentetik SLAM/VO veri seti
- **Mid-Air Dataset** — düşük irtifa drone uçuşları için çok-modlu veri seti
- **UZH-FPV Drone Racing Dataset** (Zürih Üniversitesi) — gerçek drone VO/VIO verisi
- **EuRoC MAV Dataset** — iç mekan ama VO/VIO pipeline'ını ilk kez ayağa kaldırmak için standart benchmark
- **VisDrone / UAVDT** — VO için etiketli değil ama gerçekçi hava görüntüsü çeşitliliğini (bulanıklık, farklı irtifa, farklı sahne) görmek için faydalı
- Gazebo + PX4 SITL — istersen daha gerçekçi fizik simülasyonu için

---

## Faz 7 — Kendi Değerlendirme Sistemin (Erken Kur!)

**Neden:** Denklem 2'deki hata formülünü (ortalama Öklid mesafesi) birebir kendi test scriptine yazmadan, hangi değişikliğin skoru gerçekten iyileştirdiğini bilemezsin.

**Yapman gereken:**
- Resmi formülü uygulayan bir `evaluate.py` yaz
- health=1/health=0 geçişini simüle eden sahte senaryolar kur (örn. elindeki herhangi bir sürekli videoda rastgele bir aralığı "kayıp" say, o aralıkta kendi tahminini kullan, geri kalanında gerçek değeri kullan) — böylece drift'in zamana göre nasıl büyüdüğünü gözlemleyebilirsin

---

## Faz 8 — İterasyon Planı

1. Faz 2'deki baseline'ı çalıştır → hata sayısını al
2. Faz 3'ün kalibrasyon fikrini ekle → en büyük iyileşme muhtemelen burada gelecek
3. Faz 4 filtresini ekle → drift'i düzleştir
4. Hataların hangi koşullarda (bulanık kare, düşük ışık, deniz üstü vs.) yoğunlaştığını incele
5. Sadece gerekli olan yerlerde Faz 5'e (derin öğrenme) geç — her yeri deep learning ile çözmeye çalışma, klasik yöntem zaten iyi çalışıyorsa üstüne katman ekleme

---

## Unutma

- **GitHub proje deposu** ve **Google Groups** sayfalarını takip et (Bölüm 10) — örnek veri seti, gerçek JSON formatı ve kamera parametreleri buradan gelecek.
- **Çevrimiçi Yarışma Simülasyonu** (Mayıs ayı, Bölüm 6) 1. oturumla aynı kural ve temada olacak — ikinci görev için de burada canlı test şansın olacak, bu tarihe kadar en azından Faz 1-4'ü bitirmiş olmak hedefin olsun.

---

### Bu hafta somut ilk adım
1. OpenCV kur, elindeki herhangi bir drone/hava videosunda (YouTube'dan indir ya da telefonundan çek) ORB eşleştirme + optik akış dene.
2. İki kare arası homografi çıkarıp `cv2.recoverPose` ile R,t elde et.
3. Piksel hareketini metreye çevirme problemini kâğıt üzerinde çöz — bu yarışmanın can alıcı noktası, kod yazmadan önce anlamış olman lazım.
