# AI Camera Reading Assistant

카메라 문서를 촬영해 전처리, EasyOCR, 선택적 AI 보정, TTS 순서로 읽어 주는 시각장애인 독서보조 프로그램입니다.

## 설치 및 실행

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

`config.py`에서 카메라 번호, OCR 임계값, 음성 속도와 음성을 설정할 수 있습니다. AI 보정은 프로젝트 루트의 `.env` 파일에서 설정할 수 있습니다. API 키가 없거나 호출이 실패해도 정규화한 OCR 원문을 읽습니다.

```env
# OpenAI (기본)
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4.1-mini

# Gemini 사용 시 아래처럼 지정
# AI_PROVIDER=gemini
# GEMINI_API_KEY=your_api_key
# GEMINI_MODEL=gemini-2.0-flash

# OCR 원문, 신뢰도, AI/TTS 텍스트, 처리 시간을 logs/에 자세히 기록
DEBUG_MODE=true

# 비동기 TTS: Edge-TTS를 우선 사용하고 실패하면 pyttsx3로 자동 전환
TTS_BACKEND=edge
EDGE_TTS_VOICE=ko-KR-SunHiNeural
EDGE_TTS_RATE=+0%
```

## 조작

- `SPACE`: 현재 카메라 프레임 촬영
- `ESC`: 촬영 대기 중 프로그램 종료, 음성 재생 중 즉시 중단

OpenCV 미리보기는 메인 스레드에서 계속 실행되며 OCR, Gemini, TTS는 각각 독립된 큐 워커에서 처리됩니다. 처리 중에도 창의 Windows 메시지 루프가 유지됩니다.
SPACE를 누른 순간의 프레임은 복사되어 화면에 고정되며, OCR·Gemini·TTS가 모두 끝난 뒤 READY 상태에서 실시간 미리보기가 자동으로 재개됩니다. 처리 중 ESC를 누르면 현재 결과를 폐기하고 READY로 돌아갑니다.

`captures/`, `debug/`, `logs/`는 첫 실행 시 자동으로 생성됩니다. EasyOCR는 첫 실행 시 언어 모델을 내려받을 수 있으므로 인터넷 연결이 필요할 수 있습니다.
