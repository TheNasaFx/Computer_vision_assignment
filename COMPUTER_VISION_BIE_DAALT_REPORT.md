# Компьютерийн хараа хичээлийн бие даалтын ажлын тайлан


---

## Агуулга

1. Хураангуй
2. Оршил
3. Судалгааны үндэслэл ба асуудлын тодорхойлолт
4. Төслийн зорилго, зорилтууд
5. Системийн шаардлага
6. Ашигласан технологи
7. Системийн ерөнхий архитектур
8. Backend хэрэгжүүлэлт
9. Frontend хэрэгжүүлэлт
10. Object Detection ба Tracking
11. Magic Mirror: AR Filter Mode
12. Magic Mirror: Drum Mode
13. Rhythm Game ба Beat Challenge
14. Detection тогтворжуулалтын шийдэл
15. Туршилт, шалгалт ба үр дүн
16. Төслийн давуу тал, хязгаарлалт
17. Цаашдын хөгжүүлэлт
18. Дүгнэлт
19. Ашигласан материал
20. Хавсралт

---

## 1. Хураангуй

Энэхүү бие даалтын ажлаар бид компьютерийн харааны бодит цагийн хэрэглээний
системийг веб орчинд хөгжүүлсэн. Төслийн хүрээнд webcam болон video upload
эх үүсвэрээс дүрс авч, объект илрүүлэх, объектын хүрээг тогтвортой харуулах,
нүүрний landmark ашиглан AR filter байрлуулах, гарын landmark ашиглан virtual
drum тоглуулах зэрэг хэд хэдэн компьютерийн харааны ойлголтыг нэг системд
нэгтгэсэн.

Системийн үндсэн бүрэлдэхүүн хэсгүүд нь:

- YOLO26 загвар дээр суурилсан real-time object detection
- ByteTrack tracking болон frontend temporal smoothing ашигласан тогтвортой
  bounding box visualization
- MediaPipe Face Landmarker ашигласан олон хүний нүүр дээр ажиллах AR filter
- MediaPipe Hand Landmarker ашигласан browser-side drum interaction
- Web Audio API ашигласан drum sound synthesis болон backing music
- Beat Challenge буюу score, combo, accuracy, timing judge бүхий rhythm-game
  горим
- FastAPI backend болон Next.js frontend бүхий full-stack architecture

Төслийн гол онцлог нь зөвхөн AI загвар ажиллуулах бус, model-ийн гаралтыг
хэрэглэгчийн хувьд ойлгомжтой, тогтвортой, интерактив хэлбэрээр харуулахад
чиглэсэн явдал юм. Бид detection jitter, false positive box, mirrored camera
coordinate, face filter alignment, hand tracking latency, audio feedback зэрэг
практик асуудлуудыг тус бүрд нь шийдвэрлэсэн.

---

## 2. Оршил

Компьютерийн хараа нь зураг, видео, camera stream зэрэг дүрслэлийн өгөгдлөөс
утга бүхий мэдээлэл гарган авах хиймэл оюун ухааны нэг гол салбар юм. Орчин
үеийн компьютерийн харааны системүүд нь объект илрүүлэх, хөдөлгөөн хянах,
нүүр таних, pose estimation, landmark detection, augmented reality зэрэг олон
чиглэлд өргөн хэрэглэгдэж байна.

Энэхүү бие даалтын ажлаар бид эдгээр ойлголтуудыг нэг бодит хэрэглээний
системд нэгтгэхийг зорьсон. Үүний тулд бид эхлээд object detection дээр
суурилсан live camera болон video upload detection систем хөгжүүлсэн. Дараа
нь төслийг илүү интерактив, хэрэглэгчтэй шууд харилцдаг хэлбэрт оруулахын
тулд AI Magic Mirror хэсгийг нэмсэн.

AI Magic Mirror нь хоёр үндсэн горимтой:

- **Filter Mode**: хэрэглэгчийн нүүр дээр real-time AR filter байрлуулна.
- **Drum Mode**: хэрэглэгчийн гарын хөдөлгөөнийг илрүүлж virtual drum pad
  цохих боломж олгоно.

Төслийн хөгжүүлэлтийн явцад бид computer vision model-ийн raw output-ийг
шууд хэрэглэгчид харуулах нь хангалтгүй гэдгийг анзаарсан. Жишээлбэл YOLO
объект илрүүлж байгаа боловч bounding box frame бүр дээр бага зэрэг савлах,
зарим үед нэг frame-ийн false positive box гарч ирэх, AR filter mirrored
camera дээр буруу өнцгөөр харагдах зэрэг асуудлууд гарсан. Иймээс бид model
inference-ээс гадна tracking, smoothing, coordinate correction, rendering,
audio scheduling зэрэг engineering давхаргуудыг системтэйгээр хэрэгжүүлсэн.

---

## 3. Судалгааны үндэслэл ба асуудлын тодорхойлолт

### 3.1 Судалгааны үндэслэл

Real-time computer vision application хөгжүүлэхэд дараах үндсэн сорилтууд
байдаг.

- Model inference хугацаа бага байх ёстой.
- Camera frame болон overlay хооронд latency бага байх шаардлагатай.
- Detection box frame бүр дээр савлахгүй, хэрэглэгчид тогтвортой харагдах
  хэрэгтэй.
- Нүүр болон гарын landmark-ийг дэлгэцийн coordinate system-тэй зөв тааруулах
  шаардлагатай.
- User interaction буюу гарын хөдөлгөөн, audio feedback, UI effect зэрэг нь
  синхрон ажиллах ёстой.

Бидний төсөл эдгээр асуудлуудыг нэг дор туршиж, шийдвэрлэх зорилготой.

### 3.2 Асуудлын тодорхойлолт

Системийн эхний хувилбар object detection хийж байсан боловч дараах сул талууд
илэрсэн.

- Bounding box тогтворгүй хөдөлж харагдах
- Зарим үед худлаа илүү box түр гарч ирэх
- Filter mode дээр зарим filter доошоо харсан мэт буруу байрлах
- Filter зөвхөн нэг хүний нүүр дээр ажиллах
- Drum mode дээр pad-ууд хэт том харагдах
- Drum challenge нь жинхэнэ ая, backing track-гүй тул тоглоомын мэдрэмж сул
  байх
- Challenge дуусахад үр дүнгийн feedback хангалтгүй байх

Эдгээр асуудлыг шийдвэрлэхийн тулд бид дараах чиглэлээр сайжруулалт хийсэн.

- ByteTrack болон frontend stabilizer ашиглан detection тогтворжуулах
- Face angle calculation-ийг mirrored camera view-д нийцүүлэн засах
- Face Landmarker-ийг олон хүний нүүр илрүүлэхээр тохируулах
- Drum kit-ийг 6 sound, жижиг pad layout, backing music-тэй болгох
- Challenge төгсгөлд score, accuracy болон celebration effect нэмэх

---

## 4. Төслийн зорилго, зорилтууд

### 4.1 Ерөнхий зорилго

Бидний зорилго нь computer vision model-үүдийг бодит цагийн веб application
дотор нэгтгэж, объект илрүүлэх, AR filter, гарын хөдөлгөөнөөр удирдах drum
interaction зэрэг хэрэглэгчид шууд мэдрэгдэх боломжтой систем хөгжүүлэх юм.

### 4.2 Тусгай зорилтууд

Төслийн хүрээнд дараах зорилтуудыг тавьсан.

1. Webcam болон video upload дээр YOLO object detection ажиллуулах.
2. Detection үр дүнг canvas overlay хэлбэрээр харуулах.
3. Detection jitter болон false positive box-ийг багасгах.
4. Face landmark ашиглан AR filter-үүдийг нүүр дээр зөв байрлуулах.
5. Хоёр хүний нүүр дээр filter зэрэг ажиллуулах.
6. Hand landmark ашиглан virtual drum pad цохих боломж үүсгэх.
7. Drum sound-уудыг Web Audio API ашиглан browser дээр synthesize хийх.
8. Drum mode дээр backing music бүхий rhythm challenge нэмэх.
9. Score, combo, accuracy, timing judge болон celebration effect хэрэгжүүлэх.
10. Бүх хэсгийг нэг frontend/backend architecture-д нэгтгэх.

---

## 5. Системийн шаардлага

### 5.1 Функциональ шаардлага

Систем дараах үйлдлүүдийг хийх ёстой.

- Camera stream авах
- Upload video-г унших
- Frame capture хийж backend рүү илгээх
- YOLO detection үр дүнг хүлээн авах
- Bounding box, label, confidence харуулах
- AR filter сонгох ба нүүр дээр байрлуулах
- Гар илрүүлэх ба virtual drum pad цохих
- Drum sound real-time тоглуулах
- Beat challenge эхлүүлэх, дуусгах, оноо тооцох
- Challenge төгсгөлд visual effect харуулах

### 5.2 Функциональ бус шаардлага

- Real-time ажиллагаатай байх
- Frontend video playback гацахгүй байх
- Detection overlay тогтвортой харагдах
- Browser дээр ажиллахад нэмэлт native application шаардахгүй байх
- Backend болон frontend тусдаа ажиллах боломжтой байх
- Code structure ойлгомжтой, өргөтгөхөд боломжтой байх

---

## 6. Ашигласан технологи

### 6.1 Backend технологи

| Технологи | Үүрэг |
|---|---|
| Python | Backend logic болон computer vision processing |
| FastAPI | REST API endpoint үүсгэх |
| OpenCV | Frame decode/encode, image processing |
| NumPy | Image array боловсруулах |
| Ultralytics YOLO | Object detection болон tracking |
| ByteTrack | Object identity-г frame хооронд хадгалах |

### 6.2 Frontend технологи

| Технологи | Үүрэг |
|---|---|
| Next.js | Web application framework |
| React | UI state болон component logic |
| TypeScript | Type safety |
| Tailwind CSS | Responsive UI дизайн |
| Canvas 2D API | Overlay, filter, drum pad, visual effect |
| MediaPipe Tasks Vision | Face болон hand landmark detection |
| Web Audio API | Drum sound болон backing music synthesis |

### 6.3 Computer Vision аргачлалууд

- Object Detection
- Multi-object Tracking
- Face Landmark Detection
- Hand Landmark Detection
- Temporal Smoothing
- Predictive Hit Detection
- Coordinate Transformation
- Real-time Canvas Rendering

---

## 7. Системийн ерөнхий архитектур

Системийг frontend болон backend гэсэн хоёр үндсэн хэсэгтэйгээр зохион
байгуулсан.

```text
Browser / Next.js Frontend
  ├─ Camera / Video frame capture
  ├─ Canvas rendering
  ├─ AR filter rendering
  ├─ Drum interaction
  └─ HTTP request to backend

FastAPI Backend
  ├─ JPEG frame decode
  ├─ YOLO inference
  ├─ ByteTrack tracking
  ├─ Detection metadata response
  └─ Optional annotated output
```

Frontend нь хэрэглэгчийн camera/video-г авч, frame-ийг backend рүү илгээнэ.
Backend нь тухайн frame дээр detection ажиллуулж metadata буцаана. Frontend
нь metadata-г canvas дээр зурж харуулна.

Magic Mirror-ийн Filter Mode болон Drum Mode-ийн гол inference нь browser-side
MediaPipe дээр ажилладаг. Энэ нь latency багасгах, server load бууруулах
давуу талтай.

---

## 8. Backend хэрэгжүүлэлт

Backend нь `api/server.py` файлд FastAPI application хэлбэрээр хэрэгжсэн.

### 8.1 `/detect-frame` endpoint

Энэ endpoint нь live camera болон video upload detection-д ашиглагдана.

Ажиллах дараалал:

1. Frontend frame-ийг JPEG blob хэлбэрээр илгээнэ.
2. Backend `UploadFile` уншина.
3. NumPy buffer үүсгэнэ.
4. OpenCV `imdecode` ашиглан BGR frame болгоно.
5. YOLO detection эсвэл tracking ажиллуулна.
6. Detection metadata-г JSON болгон response header-д буцаана.

### 8.2 Tracking integration

Анхны хувилбарт `/detect-frame` endpoint raw detection ашиглаж байсан. Бид
үүнийг ByteTrack ашиглах боломжтой болгосон. Ингэснээр `track_id` үүсэж,
frontend дээр detection-уудыг frame хооронд илүү найдвартай холбох боломжтой
болсон.

### 8.3 `/pose-frame` endpoint

Magic Mirror-ийн зарим pose-related туршилтад зориулж lightweight pose endpoint
нэмсэн. Энэ endpoint нь бүтэн annotated image буцаахгүй, зөвхөн keypoint
metadata буцаах зарчимтай.

---

## 9. Frontend хэрэгжүүлэлт

Frontend нь `web/app` дотор Next.js App Router бүтэцтэй.

Үндсэн page-үүд:

- `/camera`: Live camera detection
- `/demo`: Video upload detection
- `/magic-mirror`: AR filter болон drum mode
- `/study-space`: Study space monitoring туршилтын хэсэг

### 9.1 Dual-loop rendering

Camera болон video detection дээр бид dual-loop approach ашигласан.

- Display loop: `requestAnimationFrame` ашиглан video болон overlay зурна.
- Detection loop: backend рүү frame илгээж detection metadata авна.

Энэ бүтэц нь video playback-ийг detection inference-ээс тусгаарлаж, хэрэглэгчид
илүү smooth мэдрэмж өгдөг.

### 9.2 Canvas overlay

Canvas дээр дараах зүйлсийг зурдаг.

- Bounding box
- Object label болон confidence
- AR filter
- Drum pad
- Fingertip cursor
- Hit splash
- Challenge lane
- Fireworks celebration effect

Canvas 2D API нь real-time visualization хийхэд хангалттай хурдан бөгөөд
хэрэгжүүлэхэд уян хатан байсан.

---

## 10. Object Detection ба Tracking

### 10.1 YOLO object detection

YOLO model нь frame дээрх объектуудыг bounding box хэлбэрээр илрүүлнэ.
Detection бүр дараах мэдээлэлтэй.

- `bbox`: `[x1, y1, x2, y2]`
- `confidence`: model-ийн итгэлцүүр
- `class_id`: class дугаар
- `class_name`: class нэр
- `track_id`: tracking асаалттай үед object ID

### 10.2 Detection jitter асуудал

Real-time detection-д model нэг объект дээр frame бүрт бага зэрэг өөр bbox
гаргах нь түгээмэл. Энэ нь model буруу гэсэн үг биш боловч UI дээр box
савлаж харагддаг.

Мөн бага confidence бүхий false positive detection нэг frame дээр гарч ирээд
алга болох тохиолдол байдаг. Хэрэглэгчийн хувьд энэ нь тогтворгүй, алдаатай
мэт харагддаг.

### 10.3 Temporal stabilizer

Энэ асуудлыг шийдэхийн тулд бид frontend талд `DetectionStabilizer` class
хэрэгжүүлсэн. Энэ class дараах logic-ийг ашиглана.

- Track ID байгаа бол track ID-аар match хийх
- Track ID байхгүй бол IoU ашиглан өмнөх box-той match хийх
- Bounding box coordinate дээр EMA smoothing хийх
- Нэг frame-ийн сул detection-ийг шууд харуулахгүй байх
- Detection түр тасарсан үед богино хугацаанд hold хийх

EMA smoothing:

```text
smoothed = previous * (1 - alpha) + current * alpha
```

Энэ шийдлийн үр дүнд live camera болон video upload detection илүү тогтвортой
харагдах болсон.

---

## 11. Magic Mirror: AR Filter Mode

### 11.1 Face Landmarker

Filter Mode дээр бид MediaPipe Face Landmarker ашигласан. Face Landmarker нь
нүүрний олон landmark point илрүүлж, эдгээр point-оор filter-ийн байрлал,
хэмжээ, өнцгийг тодорхойлох боломж өгдөг.

Хэрэгжүүлсэн filter-үүд:

- Sunglasses
- Party Hat
- Mustache
- Devil Horns
- Dog/Pup Filter

### 11.2 Landmark geometry

Filter-ийг зөв байрлуулахын тулд бид дараах reference point-уудыг ашигласан.

- Eye center
- Eye distance
- Nose tip
- Forehead
- Chin
- Upper lip

Жишээлбэл sunglasses filter нь хоёр нүдний хоорондын зайгаар scale хийнэ.
Party hat нь forehead landmark дээр байрлана. Mustache нь nose tip болон upper
lip-ийн хооронд байрлана.

### 11.3 Mirrored camera correction

Selfie camera view дээр video-г mirror хийдэг тул face landmark-ийн x
coordinate-г canvas дээр зөв хувиргах шаардлагатай. Анхны хувилбарт party hat
болон devil horn доошоо харсан мэт харагдаж байсан. Бид eye angle тооцохдоо
screen-left eye болон screen-right eye-ийг coordinate-аар ялгаж, roll angle-ийг
зөв чиглэлээр тооцдог болгосон.

### 11.4 Multi-face filter

Face Landmarker-ийн `numFaces` тохиргоог 2 болгож, илэрсэн бүх face landmarks
дээр filter зурдаг болгосон. Ингэснээр хоёр хүн camera дээр зэрэг орсон үед
filter хоёр нүүр дээр зэрэг ажиллана.

---

## 12. Magic Mirror: Drum Mode

### 12.1 Hand Landmarker

Drum Mode дээр бид MediaPipe Hand Landmarker ашигласан. Энэ model нь нэг гар
дээр 21 landmark илрүүлдэг. Бид index fingertip болон wrist point-ийг голчлон
ашигласан.

### 12.2 Virtual drum pad

Анхны хувилбар 4 том pad-тай байсан. Бид pad layout-ийг шинэчилж, 6 pad бүхий
жижиг drum kit хэлбэртэй болгосон.

Одоогийн pad-ууд:

- Kick
- Snare
- Hi-Hat
- Crash
- Tom
- Clap

Pad-уудыг дэлгэцийн доод болон дунд хэсэгт drum kit шиг байрлуулсан. Энэ нь
гарын хөдөлгөөнөөр тоглоход илүү natural бөгөөд дэлгэцийн талбайг бага эзэлдэг.

### 12.3 Predictive hit detection

Pad дотор fingertip орсон эсэхээр шууд hit гэж үзвэл false positive их гардаг.
Иймээс бид velocity reversal дээр тулгуурласан hit detection ашигласан.

Алгоритмын ерөнхий санаа:

1. Сүүлийн хэдэн fingertip coordinate-г buffer-д хадгална.
2. Velocity болон acceleration тооцно.
3. Хурдан доош хөдөлж байгаад огцом удаашрах үед hit candidate гэж үзнэ.
4. Хөдөлгөөн хэт хэвтээ бол hit гэж үзэхгүй.
5. Hit position-ийг latency нөхөх зорилгоор бага зэрэг forward extrapolate
   хийж pad collision шалгана.

Энэ арга нь drum цохилтыг илүү бодит мэдрэмжтэй болгосон.

---

## 13. Rhythm Game ба Beat Challenge

### 13.1 Challenge-ийн зорилго

Зөвхөн virtual drum pad цохих боломжтой байх нь demo-ийн хувьд хангалттай
боловч тоглоомын зорилго сул байсан. Тиймээс бид Drum Mode дээр Rhythm Game
маягийн Beat Challenge нэмсэн.

Challenge нь хэрэглэгчид note дараалал харуулж, тухайн дарааллын дагуу зөв pad
цохих даалгавар өгнө.

### 13.2 Backing music

Challenge эхлэхэд Web Audio API ашигласан backing music явдаг. Энэ нь дараах
давхаргуудтай.

- Bass line
- Chord pad
- Light hi-hat groove
- Low-volume guide drum hits

Ингэснээр хэрэглэгч зүгээр random note дагах биш, бодит groove дээр тоглож
байгаа мэт мэдрэмж авдаг.

### 13.3 Timing judge

Hit-ийн timing-ийг дараах байдлаар үнэлнэ.

| Үнэлгээ | Цагийн зөрүү |
|---|---|
| Perfect | 75 ms хүртэл |
| Great | 140 ms хүртэл |
| Good | 220 ms хүртэл |
| Miss | 220 ms-ээс их |

Score нь timing quality болон combo multiplier-ээс хамаарч өснө.

### 13.4 Result ба celebration effect

Challenge дуусахад score, accuracy, result message canvas дээр гарна. Хэрэв
accuracy 50%-аас дээш бол жижиг fireworks celebration effect гарна. Энэ нь
тоглоомын төгсгөлийг илүү тодорхой, сонирхолтой болгосон.

---

## 14. Detection тогтворжуулалтын шийдэл

Тогтворжуулалтын шийдэл нь backend tracking болон frontend smoothing гэсэн
хоёр түвшинд хэрэгжсэн.

### 14.1 Backend түвшин

Backend дээр YOLO tracking ашиглаж object бүрт `track_id` үүсгэнэ. Track ID
нь тухайн объект frame хооронд ижил объект мөн эсэхийг тодорхойлоход ашиглагдана.

### 14.2 Frontend түвшин

Frontend stabilizer нь track ID болон IoU ашиглан detection-уудыг өмнөх frame
дээрх detection-уудтай холбодог. Дараа нь box coordinate дээр smoothing хийж
зурагддаг.

### 14.3 Үр дүн

Энэ шийдлийн дараа:

- Bounding box-ийн савалгаа багассан
- Түр хугацааны false positive box багассан
- Detection overlay илүү мэргэжлийн харагдах болсон
- Live camera болон video upload хоёуланд ижил stabilizer ашиглагдсан

---

## 15. Туршилт, шалгалт ба үр дүн

### 15.1 Build шалгалт

Frontend build:

```bash
cd Biydaalt/web
npm run build
```

Backend syntax check:

```bash
python -m py_compile api/server.py
```

Эдгээр шалгалтууд амжилттай болсон.

### 15.2 Live camera test

Live camera дээр систем object detection хийж, detection box-ууд canvas дээр
харагдсан. Stabilizer нэмсний дараа box-ууд илүү тогтвортой болсон.

### 15.3 Video upload test

Video upload хэсэгт video playback болон detection background loop тусдаа
ажилласан. Энэ нь video-г гацалт багатай тоглуулах боломж өгсөн.

### 15.4 Filter mode test

Filter mode дээр дараах үр дүн гарсан.

- Sunglasses зөв байрласан
- Party hat дээшээ зөв харсан
- Devil horns зөв байрласан
- Хоёр хүний нүүр дээр filter зэрэг ажилласан

### 15.5 Drum mode test

Drum mode дээр дараах үр дүн гарсан.

- 6 drum sound ажилласан
- Pad layout жижиг, илүү зохион байгуулалттай болсон
- Beat Challenge эхэлж note дараалал харуулсан
- Backing music явсан
- Score, combo, accuracy тооцогдсон
- Accuracy 50%-аас дээш үед fireworks effect гарсан

---

## 16. Төслийн давуу тал, хязгаарлалт

### 16.1 Давуу тал

- Real-time object detection болон interaction нэг системд нэгтгэгдсэн
- Browser-side landmark detection ашигласнаар latency багассан
- Detection overlay тогтворжуулах тусдаа logic хэрэгжсэн
- AR filter олон хүний нүүр дээр зэрэг ажиллах боломжтой болсон
- Drum mode нь backing music, score, combo, effect бүхий тоглоомын хэлбэртэй
  болсон
- Backend болон frontend салангид тул өргөтгөхөд боломжтой

### 16.2 Хязгаарлалт

- Model inference хурд төхөөрөмжийн GPU/CPU хүчин чадлаас хамаарна.
- Browser camera permission шаардлагатай.
- Lighting муу үед face/hand landmark detection чанар буурч болно.
- Drum hit detection нь хэрэглэгчийн гарын хурд, camera FPS-ээс хамаарна.
- Backing music нь synthesizer дээр суурилсан тул бүрэн professional audio
  track биш, demo-oriented guide music юм.

---

## 17. Цаашдын хөгжүүлэлт

Цаашид дараах сайжруулалтыг хийх боломжтой.

- Drum challenge дээр difficulty level нэмэх
- Custom song/chart upload хийх
- Илүү олон AR filter нэмэх
- Mobile responsive drum layout сайжруулах
- User performance history хадгалах
- Detection stabilizer parameter-үүдийг UI дээрээс тохируулах
- Study Space module-ийн alert dashboard сайжруулах
- PDF report болон demo video автоматаар generate хийх

---

## 18. Дүгнэлт

Энэхүү бие даалтын ажлаар бид real-time computer vision application хөгжүүлж,
object detection, tracking, face landmark detection, hand landmark detection,
AR overlay, audio synthesis болон rhythm-game interaction зэрэг олон ойлголтыг
нэг системд нэгтгэсэн.

Төслийн эхний зорилго нь camera болон video дээр object detection ажиллуулах
байсан. Хөгжүүлэлтийн явцад бид хэрэглэгчийн experience-ийг сайжруулахын тулд
detection stabilizer, multi-face AR filter, virtual drum kit, backing music,
score system, celebration effect зэрэг нэмэлтүүдийг хийсэн.

Үүний үр дүнд систем нь зөвхөн model-ийн үр дүн харуулах demo биш, харин
хэрэглэгчтэй бодит цагт харилцдаг interactive computer vision application
болсон. Бид энэ ажлаар computer vision model-ийг практик application-д
ашиглахдаа model inference-ээс гадна rendering, tracking, latency, smoothing,
audio feedback зэрэг инженерийн олон асуудлыг хамтад нь шийдвэрлэх шаардлагатай
гэдгийг харуулсан.

---

## 19. Ашигласан материал

1. Ultralytics. YOLO Documentation. https://docs.ultralytics.com/
2. Google. MediaPipe Tasks Vision Documentation. https://ai.google.dev/edge/mediapipe/solutions/vision
3. FastAPI Documentation. https://fastapi.tiangolo.com/
4. OpenCV Documentation. https://docs.opencv.org/
5. MDN Web Docs. Canvas API. https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API
6. MDN Web Docs. Web Audio API. https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API
7. React Documentation. https://react.dev/
8. Next.js Documentation. https://nextjs.org/docs
9. ByteTrack: Multi-Object Tracking by Associating Every Detection Box.
10. COCO Dataset and Object Detection Evaluation. https://cocodataset.org/

---

## 20. Хавсралт

### 20.1 Төслийн ажиллуулах заавар

Backend ажиллуулах:

```bash
cd Biydaalt
python cli.py serve --port 8000
```

Frontend ажиллуулах:

```bash
cd Biydaalt/web
npm run dev
```

Browser дээр нээх:

```text
http://localhost:3000
http://localhost:3000/camera
http://localhost:3000/demo
http://localhost:3000/magic-mirror
http://localhost:3000/study-space
```

### 20.2 Гол файлууд

| Файл | Үүрэг |
|---|---|
| `api/server.py` | FastAPI endpoint-үүд |
| `core/detector.py` | YOLO detector wrapper |
| `web/app/camera/page.tsx` | Live camera detection |
| `web/app/demo/page.tsx` | Video upload detection |
| `web/app/magic-mirror/page.tsx` | Magic Mirror UI |
| `web/lib/face-mesh.ts` | Face landmark wrapper |
| `web/lib/hands.ts` | Hand landmark wrapper |
| `web/lib/audio.ts` | Drum болон guide music synthesis |
| `web/lib/detection-stabilizer.ts` | Detection smoothing |

### 20.3 Build шалгах команд

```bash
cd Biydaalt/web
npm run build
```
