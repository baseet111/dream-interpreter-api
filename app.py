import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# إعدادات النموذج
MODEL_NAME = "baseet/Holomai-Dream-Model"
HF_TOKEN = "hf_PurgeraWpxR0gUCYFgccKDnFqTFneYSCSw"

# متغيرات النموذج
tokenizer = None
model = None

def load_model():
    """تحميل النموذج والتوكينايزر"""
    global tokenizer, model
    
    try:
        logger.info(f"🔄 تحميل النموذج: {MODEL_NAME}")
        
        # تحميل التوكينايزر
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            token=HF_TOKEN,
            trust_remote_code=True
        )
        
        # إعداد pad_token إذا لم يكن موجوداً
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # تحميل النموذج
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            model = model.to(device)
        
        logger.info(f"✅ تم تحميل النموذج بنجاح على {device}")
        return True
        
    except Exception as e:
        logger.error(f"❌ خطأ في تحميل النموذج: {str(e)}")
        return False

def test_model(text):
    """دالة تفسير الأحلام"""
    try:
        if model is None or tokenizer is None:
            return "❌ النموذج غير محمل"
        
        # ترميز النص
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        # نقل إلى نفس جهاز النموذج
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # توليد الاستجابة
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # فك ترميز النتيجة
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # إزالة النص الأصلي
        if text in result:
            result = result.replace(text, "").strip()
        
        return result if result else "تم توليد التفسير بنجاح"
        
    except Exception as e:
        logger.error(f"خطأ في توليد التفسير: {str(e)}")
        return f"❌ خطأ في المعالجة: {str(e)}"

@app.route('/')
def home():
    """الصفحة الرئيسية"""
    return jsonify({
        "status": "ok",
        "message": "🌙 خادم تفسير الأحلام يعمل على Railway",
        "model": MODEL_NAME,
        "model_loaded": model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })

@app.route('/health')
def health():
    """فحص حالة الخادم"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """تفسير الحلم"""
    try:
        # التحقق من تحميل النموذج
        if model is None or tokenizer is None:
            return jsonify({
                "error": "النموذج غير محمل حالياً. يرجى المحاولة بعد قليل."
            }), 503
        
        # الحصول على البيانات
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "يرجى إرسال النص في حقل 'text'"
            }), 400
        
        dream_text = data['text'].strip()
        
        # التحقق من طول النص
        if len(dream_text) < 5:
            return jsonify({
                "error": "يرجى كتابة وصف أكثر تفصيلاً للحلم"
            }), 400
        
        logger.info(f"🔮 معالجة طلب تفسير: {dream_text[:50]}...")
        
        # توليد التفسير
        interpretation = test_model(dream_text)
        
        return jsonify({
            "result": interpretation,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"خطأ في المعالجة: {str(e)}")
        return jsonify({
            "error": f"حدث خطأ في المعالجة: {str(e)}"
        }), 500

if __name__ == '__main__':
    # تحميل النموذج عند بدء التشغيل
    logger.info("🚀 بدء تشغيل خادم تفسير الأحلام على Railway...")
    
    # محاولة تحميل النموذج
    model_loaded = load_model()
    if not model_loaded:
        logger.warning("⚠️ فشل في تحميل النموذج عند البدء")
    
    # الحصول على البورت من متغير البيئة
    port = int(os.environ.get('PORT', 8000))
    
    # تشغيل الخادم
    app.run(host='0.0.0.0', port=port, debug=False)