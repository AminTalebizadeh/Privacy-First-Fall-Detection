#!/usr/bin/env python3
"""Export a full Persian teaching guide for the fall-detection project as .docx."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

OUT = Path(__file__).resolve().parents[1] / "docs" / "Project_Guide_FA.docx"
FONT = "Tahoma"  # widely available on Windows; supports Persian


def set_run_font(run, size=11, bold=False, code=False):
    run.font.name = "Consolas" if code else FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    run.font.size = Pt(size)
    run.bold = bold
    if code:
        run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)


def set_rtl(paragraph):
    pPr = paragraph._p.get_or_add_pPr()
    bidi = OxmlElement("w:bidi")
    bidi.set(qn("w:val"), "1")
    pPr.append(bidi)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    pf = paragraph.paragraph_format
    pf.space_after = Pt(8)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE


def add_heading_fa(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    set_rtl(h)
    for run in h.runs:
        set_run_font(run, size={1: 18, 2: 14, 3: 12}.get(level, 12), bold=True)
    return h


def add_p(doc, text, size=11, bold=False):
    p = doc.add_paragraph()
    set_rtl(p)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold)
    return p


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        set_rtl(p)
        run = p.add_run(item)
        set_run_font(run, size=11)


def add_code(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_before = Pt(4)
    pf.space_after = Pt(8)
    run = p.add_run(text)
    set_run_font(run, size=9, code=True)
    return p


def build() -> Path:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)

    # Title
    t = doc.add_paragraph()
    set_rtl(t)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("راهنمای کامل آموزشی پروژه")
    set_run_font(r, size=20, bold=True)

    t2 = doc.add_paragraph()
    set_rtl(t2)
    t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = t2.add_run(
        "Privacy-Preserving Fall Detection from Skeleton Dynamics\n"
        "تشخیص سقوط با حفظ حریم خصوصی بر اساس اسکلت انسان"
    )
    set_run_font(r2, size=13, bold=True)

    add_p(
        doc,
        "این سند برای تسلط کامل بر پروژه نوشته شده است تا بتوانید آن را "
        "برای کارفرما توضیح دهید. توضیحات به فارسی است؛ نام فایل‌ها، کلاس‌ها، "
        "توابع و قطعات کد به همان صورت انگلیسی پروژه باقی مانده‌اند.",
    )

    # ------------------------------------------------------------------
    add_heading_fa(doc, "۱. ایده اصلی پروژه (داستان محصول)", 1)
    add_p(
        doc,
        "هدف محصول این است که سقوط انسان را به‌صورت بلادرنگ تشخیص دهد، "
        "اما بدون ذخیره یا نمایش ویدیوی خام. به زبان ساده: دوربین فقط برای "
        "لحظه‌ای اسکلت بدن را استخراج می‌کند؛ بعد از آن پیکسل‌ها دور ریخته "
        "می‌شوند و کل تصمیم‌گیری فقط روی مختصات مفاصل (keypoints) انجام می‌شود.",
    )
    add_p(doc, "چرا این رویکرد برای کارفرما مهم است؟", bold=True)
    add_bullets(
        doc,
        [
            "حریم خصوصی: تصویر چهره، لباس، اتاق و جزئیات خانه ذخیره نمی‌شود.",
            "سبک بودن: مدل زمانی بسیار کوچک است (حدود ۵۰ هزار پارامتر برای GRU).",
            "قابل دفاع فنی: دو لایه محافظت حریم خصوصی در تست و ممیزی کد وجود دارد.",
            "قابل توضیح بودن: ویژگی‌هایی مثل زاویه تنه و سرعت لگن معنای فیزیکی دارند.",
        ],
    )

    add_heading_fa(doc, "۲. نمای کلی خط لوله (Pipeline)", 1)
    add_p(doc, "کل سیستم را مثل یک خط تولید تصور کنید:")
    add_bullets(
        doc,
        [
            "ورودی: فریم ویدیو یا وب‌کم (فقط در RAM).",
            "استخراج اسکلت با YOLOv8n-pose → ۱۷ نقطه بدن برای هر شخص.",
            "ردیابی افراد با IoU/Centroid تا هویت کوتاه‌مدت پایدار بماند.",
            "نرمال‌سازی و ساخت ویژگی‌های زمانی → بردار ۵۵‌بُعدی در هر فریم.",
            "پنجره‌های لغزان ۳۰ فریمی (حدود ۱ ثانیه) با گام ۵.",
            "طبقه‌بند زمانی (GRU یا 1D-CNN) → احتمال سقوط p(fall).",
            "مدیر هشدار با debounce و cooldown → ثبت JSON بدون تصویر.",
        ],
    )
    add_code(
        doc,
        "video -> PoseExtractor -> Tracker -> Features(T,55)\n"
        "     -> Window(30) -> GRU/CNN -> AlertManager -> alerts.jsonl",
    )

    # ------------------------------------------------------------------
    add_heading_fa(doc, "۳. ساختار پوشه‌های پروژه", 1)
    add_bullets(
        doc,
        [
            "configs/ : تنظیمات متمرکز (default.yaml و smoke.yaml).",
            "src/pose/ : استخراج اسکلت و ردیابی.",
            "src/features/ : نرمال‌سازی و ویژگی‌های حرکتی.",
            "src/model/ : شبکه‌های GRU و CNN و Focal Loss.",
            "src/data/ : دیتاست پنجره‌ها و تقسیم گروهی.",
            "src/realtime/ : سیستم هشدار حریم‌خصوصی‌محور.",
            "scripts/ : اجرای عملیاتی: کش، آموزش، ارزیابی، دمو، ممیزی.",
            "tests/ : تست‌های واحد برای ویژگی‌ها، ردیاب و حریم خصوصی.",
            "cache/skeletons/ : کش اسکلت (داده آموزش بدون پیکسل).",
            "checkpoints/ : وزن‌های بهترین مدل.",
            "data/ : برچسب‌ها و ویدیوها.",
        ],
    )

    # ------------------------------------------------------------------
    add_heading_fa(doc, "۴. فایل تنظیمات configs/default.yaml", 1)
    add_p(
        doc,
        "این فایل «منبع حقیقت» تمام پارامترهای قابل تنظیم است. دلیل جدا بودن "
        "تنظیمات از کد این است که بدون دست‌کاری منطق برنامه بتوانید آستانه "
        "هشدار، اندازه پنجره، نرخ یادگیری و غیره را عوض کنید.",
    )
    add_p(doc, "بخش‌های مهم:", bold=True)
    add_bullets(
        doc,
        [
            "pose: مدل YOLO، اندازه ورودی ۶۴۰، دستگاه cuda/cpu، دقت FP16.",
            "tracker: آستانه IoU، فاصله centroid، max_age، min_hits.",
            "features: حداقل اعتماد مفصل، fps اسمی برای سرعت.",
            "window: size=30، stride=5، positive_overlap=0.30.",
            "model: نوع gru/cnn و ابعاد شبکه.",
            "train: batch، epochs، lr، focal loss، early stopping.",
            "alerts: threshold، K consecutive، cooldown، مسیر لاگ.",
            "data: مسیر کش و labels.csv و نسبت split گروهی.",
            "realtime: حالت privacy و طول sparkline.",
        ],
    )
    add_p(
        doc,
        "فایل configs/smoke.yaml برای آزمایش سریع محلی (مثلاً روی CPU و با "
        "داده مصنوعی) اضافه شده است و منطق اصلی محصول را تغییر نمی‌دهد.",
    )

    # ------------------------------------------------------------------
    add_heading_fa(doc, "۵. ماژول Pose — استخراج اسکلت", 1)
    add_heading_fa(doc, "۵.۱ کلاس PoseExtractor (src/pose/extractor.py)", 2)
    add_p(
        doc,
        "این کلاس تنها جایی است که پیکسل خام واقعاً وارد مدل بینایی می‌شود. "
        "مسئولیتش فقط inference اسکلت است، نه ذخیره‌سازی تصویر.",
    )
    add_p(doc, "چرا این کلاس نوشته شده؟", bold=True)
    add_p(
        doc,
        "تا وابستگی به ultralytics پشت یک واسط تمیز پنهان شود؛ تست‌های ویژگی "
        "بتوانند بدون نصب YOLO اجرا شوند (import تنبل/lazy)؛ و تضمین حریم "
        "خصوصی از همین لایه شروع شود.",
    )
    add_p(doc, "__init__(model, imgsz, conf, device, half)", bold=True)
    add_bullets(
        doc,
        [
            "مدل YOLOv8n-pose را بارگذاری می‌کند.",
            "اگر CUDA نباشد، خودکار به CPU می‌رود و FP16 را خاموش می‌کند.",
            "یک warm-up با فریم صفر انجام می‌دهد تا اولین فریم واقعی کند نباشد.",
            "پارامتر half به quantize=16 نگاشت می‌شود (سازگاری با ultralytics جدید).",
        ],
    )
    add_p(doc, "__call__(frame) → boxes, kpts", bold=True)
    add_bullets(
        doc,
        [
            "روی یک فریم BGR پیش‌بینی می‌کند.",
            "boxes به شکل (N,4) در مختصات xyxy.",
            "kpts به شکل (N,17,3) شامل x و y و confidence هر مفصل.",
            "اگر کسی تشخیص داده نشود، آرایه‌های خالی برمی‌گرداند تا بقیه سیستم نشکند.",
        ],
    )

    add_heading_fa(doc, "۵.۲ تابع iter_video_frames", 2)
    add_p(
        doc,
        "این تابع فریم‌ها را یکی‌یکی از فایل ویدیو یا وب‌کم می‌خواند و yield می‌کند. "
        "دلیل طراحی generator بودن آن این است که فریم‌ها بافر یا ذخیره نشوند؛ "
        "همین‌که مصرف شدند از حافظه خارج می‌شوند. پارامتر max_frames برای "
        "آزمایش دود (smoke test) بدون باز شدن GUI مفید است.",
    )

    # ------------------------------------------------------------------
    add_heading_fa(doc, "۶. ماژول Tracker — ردیابی افراد", 1)
    add_heading_fa(doc, "۶.۱ تابع iou_xyxy", 2)
    add_p(
        doc,
        "اشتراک‌بر‌اتحاد دو جعبه‌مرزی را حساب می‌کند. دلیل وجودش: معیار اصلی "
        "برای فهمیدن اینکه تشخیص جدید همان شخص قبلی است یا نه.",
    )
    add_heading_fa(doc, "۶.۲ تابع کمکی _centroid", 2)
    add_p(
        doc,
        "مرکز جعبه را می‌دهد. وقتی شکل bbox در لحظه سقوط به‌شدت عوض می‌شود، "
        "IoU ممکن است صفر شود؛ فاصله مرکزها کمک می‌کند هویت حفظ شود.",
    )
    add_heading_fa(doc, "۶.۳ کلاس Track", 2)
    add_p(
        doc,
        "یک ساختار داده برای وضعیت هر شخص ردیابی‌شده است: شناسه، آخرین bbox، "
        "آخرین keypoints، تعداد hit، سن بدون تطبیق، وضعیت confirmed و تاریخچه.",
    )
    add_p(
        doc,
        "چرا confirmed لازم است؟ تا ردیابی‌های نویزی تک‌فریم‌ای وارد طبقه‌بند نشوند. "
        "فقط بعد از min_hits تطبیق متوالی، شخص «تأییدشده» محسوب می‌شود.",
    )
    add_heading_fa(doc, "۶.۴ کلاس IoUCentroidTracker", 2)
    add_p(
        doc,
        "ردیاب سبک چندنفره. عمداً بدون Kalman و بدون ReID نوشته شده، چون برای "
        "تشخیص سقوط فقط هویت کوتاه‌مدت لازم است نه بازشناسی بلندمدت.",
    )
    add_p(doc, "متد update(boxes, keypoints)", bold=True)
    add_bullets(
        doc,
        [
            "گذر ۱: تطبیق حریصانه بر اساس بیشترین IoU بالاتر از iou_gate.",
            "گذر ۲: برای باقی‌مانده‌ها، تطبیق با فاصله centroid کمتر از centroid_gate_px.",
            "تشخیص‌های بی‌تطبیق → ساخت Track جدید (birth).",
            "Trackهایی که بیش از max_age دیده نشوند حذف می‌شوند (death).",
            "فقط Trackهای confirmed را برمی‌گرداند.",
        ],
    )
    add_p(doc, "متد داخلی _assign", bold=True)
    add_p(
        doc,
        "یک تشخیص را به یک Track موجود وصل می‌کند، bbox و keypoints را به‌روز "
        "می‌کند، hits را زیاد و age را صفر می‌کند و در صورت کافی بودن hits، "
        "confirmed را True می‌گذارد.",
    )

    # ------------------------------------------------------------------
    add_heading_fa(doc, "۷. ماژول Features — ویژگی‌های اسکلت", 1)
    add_heading_fa(doc, "۷.۱ frame_features.py", 2)
    add_p(
        doc,
        "این فایل فقط با numpy کار می‌کند و ویژگی‌های تک‌فریم و دینامیک را "
        "می‌سازد. دلیل جدا بودن آن: قابل تست بودن بدون torch و YOLO.",
    )

    add_p(doc, "ثابت‌های L_SHOULDER, R_SHOULDER, L_HIP, R_HIP", bold=True)
    add_p(
        doc,
        "ایندکس مفاصل مهم COCO-17. تنه بدن با شانه و لگن تعریف می‌شود چون "
        "نسبت به اندام‌ها صلب‌تر است.",
    )

    add_p(doc, "تابع torso_normalize", bold=True)
    add_p(
        doc,
        "مختصات پیکسلی را از موقعیت و فاصله دوربین مستقل می‌کند. مرکز لگن را "
        "صفر می‌کند (حذف انتقال) و بر طول تنه تقسیم می‌کند (حذف مقیاس). "
        "مفصل‌های کم‌اعتماد ماسک می‌شوند تا مدل صفر را به‌عنوان موقعیت واقعی "
        "یاد نگیرد. اگر تنه دیده نشود، از fallback_scale فریم قبل یا ارتفاع "
        "bbox مفاصل مرئی استفاده می‌کند تا تقسیم بر صفر رخ ندهد.",
    )

    add_p(doc, "تابع trunk_angle", bold=True)
    add_p(
        doc,
        "زاویه تنه نسبت به راستای عمودی بالا را برحسب رادیان می‌دهد. ایستاده "
        "حدود ۰ و درازکش حدود π/۲ است. این یکی از قوی‌ترین نشانه‌های سقوط است. "
        "اگر مفاصل کافی نباشند NaN برمی‌گرداند.",
    )

    add_p(doc, "تابع bbox_aspect_ratio", bold=True)
    add_p(
        doc,
        "نسبت عرض به ارتفاع اسکلت مرئی را حساب می‌کند. ایستاده معمولاً کمتر از "
        "۱ و خوابیده بیشتر از ۱ است. مکمل زاویه تنه است و از keypoints ساخته "
        "می‌شود نه از bbox شلِ دتکتور.",
    )

    add_p(doc, "تابع hip_center_y", bold=True)
    add_p(
        doc,
        "موقعیت عمودی مرکز لگن را می‌دهد. چون در مختصات تصویر y رو به پایین "
        "زیاد می‌شود، در سقوط این مقدار معمولاً افزایش می‌یابد.",
    )

    add_p(doc, "تابع finite_diff", bold=True)
    add_p(
        doc,
        "سرعت و شتاب یک سری یک‌بُعدی را با تفاضل مرکزی حساب می‌کند. چرا مرکزی؟ "
        "چون دقیق‌تر است و قله سرعت را جابه‌جا نمی‌کند. NaNها اول درون‌یابی "
        "خطی می‌شوند تا یک فریم گم‌شده سرعت بی‌نهایت نسازد. ضرب در fps واحد را "
        "به «بر ثانیه» تبدیل می‌کند تا ویدیوهای با نرخ فریم مختلف قابل مقایسه باشند.",
    )

    add_p(doc, "تابع normalized_hip_dynamics", bold=True)
    add_p(
        doc,
        "سرعت/شتاب عمودی لگن را بر طول تنه تقسیم می‌کند تا واحد "
        "«طول‌تنه بر ثانیه» شود. در نتیجه سقوط از فاصله ۲ متری و ۶ متری برای "
        "مدل شبیه هم به نظر می‌رسد.",
    )

    add_heading_fa(doc, "۷.۲ sequence.py", 2)
    add_p(doc, "ثابت FEATURE_DIM = 55", bold=True)
    add_p(doc, "چیدمان هر فریم ویژگی:")
    add_bullets(
        doc,
        [
            "[0:34] مختصات نرمال ۱۷ مفصل × ۲",
            "[34:51] ماسک اعتماد ۱۷ مفصل",
            "[51] زاویه تنه",
            "[52] نسبت ابعاد bbox اسکلت",
            "[53] سرعت عمودی لگن نرمال",
            "[54] شتاب عمودی لگن نرمال",
        ],
    )

    add_p(doc, "تابع _ffill_nan", bold=True)
    add_p(
        doc,
        "NaN را با آخرین مقدار معتبر پر می‌کند. برخلاف صفر کردن، پرش ساختگی "
        "در سری زمانی ایجاد نمی‌کند.",
    )

    add_p(doc, "تابع sequence_features", bold=True)
    add_p(
        doc,
        "کل مسیر یک شخص را از شکل (T,17,3) به ماتریس (T,55) تبدیل می‌کند. "
        "در حلقه فریم‌ها نرمال‌سازی، زاویه، aspect و hip_y را می‌سازد؛ سپس "
        "دینامیک لگن را اضافه و همه را به هم می‌چسباند. این تابع قلب "
        "representation یادگیری است.",
    )

    add_p(doc, "تابع sliding_windows", bold=True)
    add_p(
        doc,
        "پنجره‌های هم‌اندازه از ویژگی‌ها می‌سازد. size=30 تقریباً یک ثانیه در "
        "۳۰fps است؛ stride=5 تعادل بین پوشش زمانی و تعداد نمونه‌هاست. انتهای "
        "ناقص دور ریخته می‌شود چون نسبت به نویز برچسب ناچیز است.",
    )

    add_p(doc, "تابع window_label", bold=True)
    add_p(
        doc,
        "پنجره را مثبت می‌کند فقط اگر حداقل ۳۰٪ فریم‌هایش داخل بازه سقوط "
        "حاشیه‌نویسی‌شده باشد. دلیل: اگر پنجره‌ای فقط نوک سقوط را بزند ولی "
        "عمدتاً ایستاده باشد و مثبت شود، مدل یاد می‌گیرد روی افراد ایستاده "
        "هم آلارم بدهد.",
    )

    # ------------------------------------------------------------------
    add_heading_fa(doc, "۸. ماژول Data — داده و تقسیم‌بندی", 1)
    add_heading_fa(doc, "۸.۱ تابع load_labels", 2)
    add_p(
        doc,
        "فایل data/labels.csv را می‌خواند. ستون‌ها: clip_id، fall_start_frame، "
        "fall_end_frame، group. برای فعالیت روزانه (ADL) بازه سقوط برابر "
        "-1,-1 است و به None تبدیل می‌شود.",
    )
    add_heading_fa(doc, "۸.۲ تابع split_groups", 2)
    add_p(
        doc,
        "تقسیم train/val/test را روی group انجام می‌دهد نه روی کلیپ. این مهم‌ترین "
        "تصمیم روش‌شناختی پروژه است: اگر کلیپ‌های یک نفر هم در train و هم در "
        "test باشند، معیارها مصنوعی بالا می‌روند. با group-disjoint بودن، "
        "تعمیم به افراد/صحنه‌های ندیده سنجیده می‌شود.",
    )
    add_heading_fa(doc, "۸.۳ کلاس SkeletonWindowDataset", 2)
    add_p(
        doc,
        "از کش .npz ویژگی می‌سازد، پنجره می‌گیرد و برچسب می‌زند. خروجی هر نمونه "
        "یک تنسور (30,55) و یک برچسب باینری است. خاصیت pos_fraction نسبت "
        "نمونه‌های مثبت را برای گزارش و دیباگ می‌دهد.",
    )
    add_p(
        doc,
        "نکته حریم خصوصی: همین کش اسکلت «داده آموزش» است. بعد از ساخت کش، "
        "آموزش دیگر به پیکسل نیاز ندارد.",
    )

    # ------------------------------------------------------------------
    add_heading_fa(doc, "۹. ماژول Model — طبقه‌بند زمانی", 1)
    add_p(
        doc,
        "هر دو مدل عمداً کوچک‌اند (< ۱ میلیون پارامتر). چون YOLO کار سخت "
        "درک تصویر را کرده؛ اینجا فقط امضای حرکتی حدود نیم‌ثانیه تا یک ثانیه "
        "باید تشخیص داده شود.",
    )

    add_heading_fa(doc, "۹.۱ کلاس GRUClassifier", 2)
    add_p(
        doc,
        "یک GRU دولايه با hidden=64 و سپس LayerNorm + Dropout + Linear به "
        "یک logit. در forward فقط آخرین حالت پنهان استفاده می‌شود چون خلاصه "
        "کل پنجره را حمل می‌کند. GRU برای وابستگی زمانی و الگوی «افت → "
        "برخورد» مناسب است.",
    )

    add_heading_fa(doc, "۹.۲ کلاس CNN1DClassifier", 2)
    add_p(
        doc,
        "خط پایه مقایسه‌ای: چند لایه Conv1d-BN-ReLU و سپس Global Max Pool. "
        "CNN موتیف‌های محلی مثل ضربه برخورد را خوب می‌بیند ولی حافظه/گیت "
        "ندارد. اگر GRU فقط به اندازه CNN خوب باشد، یعنی بازگشت‌پذیری ارزش "
        "اضافه‌ای نداشته.",
    )

    add_heading_fa(doc, "۹.۳ کلاس FocalLoss", 2)
    add_p(
        doc,
        "چون پنجره‌های FALL معمولاً ۵ تا ۱۰ درصد داده‌اند، BCE معمولی مدل را "
        "به سمت کلاس NORMAL هل می‌دهد. Focal Loss نمونه‌های آسان منفی را "
        "کم‌وزن و با alpha تعادل کلاس را حفظ می‌کند بدون undersampling.",
    )

    add_heading_fa(doc, "۹.۴ تابع build_model", 2)
    add_p(
        doc,
        "از روی config مدل درست را می‌سازد. یک نقطه ورود واحد برای train، "
        "evaluate و realtime تا معماری همیشه با checkpoint هم‌خوان باشد.",
    )

    # ------------------------------------------------------------------
    add_heading_fa(doc, "۱۰. ماژول Realtime — هشدارها", 1)
    add_heading_fa(doc, "۱۰.۱ ثابت EVENT_SCHEMA", 2)
    add_p(
        doc,
        "اسکیمای لاگ قفل شده است: event, timestamp, track_id, confidence. "
        "هیچ فیلدی برای تصویر وجود ندارد.",
    )
    add_heading_fa(doc, "۱۰.۲ کلاس‌های _TrackAlertState و AlertEvent", 2)
    add_p(
        doc,
        "_TrackAlertState وضعیت داخلی هر track را نگه می‌دارد (تعداد مثبت "
        "متوالی، زمان آخرین آلارم، تاریخچه احتمال). AlertEvent رویداد نهایی "
        "قابل ثبت است و متد to_json آن را با اسکیمای ثابت به JSON تبدیل می‌کند.",
    )
    add_heading_fa(doc, "۱۰.۳ کلاس AlertManager", 2)
    add_p(doc, "قوانین هشدار:", bold=True)
    add_bullets(
        doc,
        [
            "پنجره مثبت است اگر p(fall) >= threshold.",
            "آلارم فقط بعد از K پنجره مثبت متوالی برای همان track شلیک می‌شود (debounce).",
            "بعد از شلیک، cooldown فعال می‌شود تا یک سقوط به ده‌ها آلارم تبدیل نشود.",
        ],
    )
    add_p(doc, "متدها:", bold=True)
    add_bullets(
        doc,
        [
            "update: یک احتمال جدید می‌گیرد؛ اگر شرط‌ها برقرار باشد AlertEvent برمی‌گرداند و در فایل می‌نویسد.",
            "history: برای رسم sparkline احتمال‌های اخیر را می‌دهد.",
            "close: فایل لاگ را می‌بندد.",
        ],
    )

    # ------------------------------------------------------------------
    add_heading_fa(doc, "۱۱. اسکریپت‌های اجرایی (از اول تا آخر)", 1)

    add_heading_fa(doc, "۱۱.۱ گام داده واقعی: آماده‌سازی ویدیو و برچسب", 2)
    add_p(
        doc,
        "برای محصول واقعی، کلیپ‌های مجموعه UR Fall Detection (یا مشابه) در "
        "data/videos قرار می‌گیرند و data/labels.csv ساخته می‌شود. group باید "
        "واحد جلوگیری از نشت باشد (مثلاً همان سوژه یا همان صحنه).",
    )

    add_heading_fa(doc, "۱۱.۲ scripts/cache_skeletons.py", 2)
    add_p(
        doc,
        "این مرحله «نمایش عملی حریم خصوصی» است: ویدیو خوانده می‌شود، اسکلت "
        "استخراج و ردیابی می‌شود، و فقط keypoints + frame_idx + fps در "
        "cache/skeletons/<clip>.npz ذخیره می‌شود. تابع "
        "primary_track_sequence طولانی‌ترین مسیر تأییدشده را برمی‌دارد "
        "(در دیتاست‌های تک‌نفره همان بازیگر است).",
    )

    add_heading_fa(doc, "۱۱.۳ scripts/make_synthetic_cache.py (کمک دیباگ)", 2)
    add_p(
        doc,
        "اگر ویدیوی واقعی نداشته باشید، اسکلت مصنوعی fall/ADL می‌سازد تا "
        "خط لوله آموزش و ارزیابی قابل اجرا باشد. این برای دموی داخلی/دیباگ "
        "است نه جایگزینی دیتاست رسمی برای ادعاهای دقت نهایی نزد کارفرما.",
    )

    add_heading_fa(doc, "۱۱.۴ scripts/train.py", 2)
    add_p(doc, "توابع کلیدی:", bold=True)
    add_bullets(
        doc,
        [
            "set_seed: تکرارپذیری آزمایش‌ها.",
            "evaluate: Precision/Recall/F1 روی validation.",
            "make_split: ساخت دیتاست یک بخش با بررسی وجود labels و cache.",
            "main: ساخت مدل، FocalLoss، AdamW، حلقه epoch، early stopping روی val-F1، ذخیره best_<model>.pt همراه config.",
        ],
    )
    add_p(
        doc,
        "چرا early stopping روی F1 است نه فقط loss؟ چون در داده نامتوازن، "
        "کاهش loss لزوماً به معنای آلارم‌های مفید نیست؛ F1 به تعادل دقت و "
        "بازیابی نزدیک‌تر است.",
    )

    add_heading_fa(doc, "۱۱.۵ scripts/evaluate.py", 2)
    add_p(doc, "سه گزارش مهم برای کارفرما:", bold=True)
    add_bullets(
        doc,
        [
            "معیار سطح پنجره: Precision/Recall/F1 روی هر window.",
            "معیار سطح رویداد: آیا آلارم در بازه ±۲ ثانیه اطراف سقوط واقعی آمده؟ این به کاربرد واقعی نزدیک‌تر است.",
            "سخت‌ترین False Positiveها: پنجره‌های منفی با بالاترین احتمال؛ معمولاً نشستن سریع یا دراز کشیدن عمدی.",
        ],
    )
    add_p(
        doc,
        "معیار پذیرش اعلام‌شده در README: Event-level F1 ≥ 0.90 و سرعت "
        "حداقل ۱۵ FPS در رزولوشن ۶۴۰ روی GPU هدف.",
    )

    add_heading_fa(doc, "۱۱.۶ scripts/run_realtime.py", 2)
    add_p(
        doc,
        "دموی بلادرنگ. مسیر کامل را روی وب‌کم یا فایل اجرا می‌کند. در حالت "
        "privacy (پیش‌فرض) اسکلت روی بوم سیاه کشیده می‌شود و ویدیوی خام "
        "نشان داده نمی‌شود. با --no-privacy برای دیباگ روی فریم اصلی overlay "
        "می‌شود، ولی باز هم چیزی ضبط نمی‌شود.",
    )
    add_p(doc, "اجزای مهم داخل main:", bold=True)
    add_bullets(
        doc,
        [
            "load_classifier: بارگذاری checkpoint و ساخت مدل از config ذخیره‌شده.",
            "بافر deque به طول ۳۰ برای هر track.",
            "طبقه‌بندی هر stride فریم وقتی بافر پر است.",
            "draw_scene و draw_sparkline برای نمایش وضعیت و روند احتمال.",
            "AlertManager برای debounce و نوشتن alerts.jsonl.",
        ],
    )

    add_heading_fa(doc, "۱۱.۷ scripts/audit_no_frame_writes.py", 2)
    add_p(
        doc,
        "ممیزی ایستای کد: اگر جایی در src یا scripts فراخوانی‌هایی مثل "
        "cv2.imwrite یا VideoWriter باشد، شکست می‌خورد. این مکمل تست اسکیمای "
        "لاگ است تا معیار «هیچ مسیری فریم خام ننویسد» در سطح سورس هم تضمین شود.",
    )

    # ------------------------------------------------------------------
    add_heading_fa(doc, "۱۲. تست‌ها و تضمین کیفیت", 1)
    add_bullets(
        doc,
        [
            "tests/test_normalize.py: ناوردایی انتقال/مقیاس، ماسک اعتماد کم، زاویه تنه ایستاده/خوابیده.",
            "tests/test_dynamics.py: صحت سرعت/شتاب، درون‌یابی گپ، علامت سرعت در سقوط مصنوعی.",
            "tests/test_alerts_and_tracker.py: پایداری هویت، مرگ track، debounce، cooldown، اسکیمای لاگ بدون تصویر.",
        ],
    )
    add_p(
        doc,
        "این تست‌ها به کارفرما نشان می‌دهند که بخش‌های حساس (ویژگی‌ها، ردیاب، "
        "حریم خصوصی) فقط «ادعا» نیستند؛ با واحدتست قفل شده‌اند.",
    )

    # ------------------------------------------------------------------
    add_heading_fa(doc, "۱۳. حریم خصوصی: چه چیزی ذخیره می‌شود و چه چیزی نه", 1)
    add_p(doc, "ذخیره می‌شود:", bold=True)
    add_bullets(
        doc,
        [
            "cache/skeletons/*.npz : مختصات و اعتماد مفاصل + اندیس فریم + fps.",
            "checkpoints/*.pt : وزن مدل + config.",
            "alerts.jsonl : فقط event/timestamp/track_id/confidence.",
        ],
    )
    add_p(doc, "هرگز ذخیره نمی‌شود:", bold=True)
    add_bullets(
        doc,
        [
            "فریم خام، کراپ، تامبنیل، یا هر داده پیکسلی.",
            "مسیر ضبط ویدیو در مخزن وجود ندارد.",
        ],
    )
    add_p(
        doc,
        "نکته صادقانه برای کارفرما: اسکلت هم داده رفتاری است و می‌تواند به "
        "فرد قابل نسبت باشد؛ سیاست نگهداری کش و لاگ موضوع استقرار است نه کد.",
    )

    # ------------------------------------------------------------------
    add_heading_fa(doc, "۱۴. ترتیب اجرای کامل (برای توضیح به کارفرما)", 1)
    add_code(
        doc,
        "1) pip install -r requirements.txt\n"
        "2) Put videos in data/videos + build data/labels.csv\n"
        "3) python scripts/cache_skeletons.py --videos data/videos\n"
        "4) python scripts/train.py --model gru\n"
        "5) python scripts/train.py --model cnn\n"
        "6) python scripts/evaluate.py --ckpt checkpoints/best_gru.pt --split test\n"
        "7) python scripts/run_realtime.py --source 0\n"
        "8) python -m pytest tests/ -v\n"
        "9) python scripts/audit_no_frame_writes.py",
    )

    # ------------------------------------------------------------------
    add_heading_fa(doc, "۱۵. چگونه برای کارفرما توضیح دهید (نسخه کوتاه ارائه)", 1)
    add_p(
        doc,
        "۱) مشکل: سامانه‌های معمولی ویدیوی خانه را ذخیره یا پردازش می‌کنند و "
        "ریسک حریم خصوصی دارند.",
    )
    add_p(
        doc,
        "۲) راه‌حل ما: فقط اسکلت استخراج می‌شود؛ پیکسل‌ها دور ریخته می‌شوند؛ "
        "تصمیم با مدل زمانی سبک روی ویژگی‌های حرکتی گرفته می‌شود.",
    )
    add_p(
        doc,
        "۳) چرا قابل اعتماد است: تقسیم گروهی بدون نشت فرد، معیار رویدادمحور، "
        "debounce برای کاهش آلارم کاذب، و ممیزی/تست حریم خصوصی.",
    )
    add_p(
        doc,
        "۴) خروجی عملیاتی: هشدار JSON با زمان و شناسه مسیر و اطمینان؛ بدون تصویر.",
    )
    add_p(
        doc,
        "۵) محدودیت‌ها/گام بعدی: نیاز به داده واقعی کافی، تنظیم آستانه برای "
        "نشستن سریع/درازکشیدن عمدی، و سیاست نگهداری لاگ در استقرار.",
    )

    # ------------------------------------------------------------------
    add_heading_fa(doc, "۱۶. جمع‌بندی ماژول‌به‌ماژول (جدول ذهنی)", 1)
    add_bullets(
        doc,
        [
            "PoseExtractor: چشم سیستم؛ پیکسل → اسکلت.",
            "IoUCentroidTracker: حافظه کوتاه‌مدت هویت افراد.",
            "torso_normalize + sequence_features: زبان مشترک حرکت مستقل از دوربین.",
            "SkeletonWindowDataset + split_groups: داده تمیز و ارزیابی صادقانه.",
            "GRU/CNN + FocalLoss: تشخیص الگوی سقوط در پنجره زمانی.",
            "AlertManager: تبدیل احتمال به هشدار قابل استفاده و کم‌نویز.",
            "scripts/*: چرخه عمر از کش تا دمو و ممیزی.",
            "tests + audit: قفل کیفیت و حریم خصوصی.",
        ],
    )

    add_p(
        doc,
        "اگر این ۱۶ بخش را بتوانید به ترتیب بالا روایت کنید، کل پروژه را "
        "از ایده تا استقرار در حد ارائه به کارفرما مسلط هستید.",
        bold=True,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote: {path}")
