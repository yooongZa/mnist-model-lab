"""streamlit run streamlit_app.py"""
from pathlib import Path
import html
import json

import pandas as pd
import altair as alt
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from mnist_lab.inference import load_model_bytes, predict, prepare_drawing
from mnist_lab.sequential import EXAMPLE_CODE, LAYER_NAMES

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="MNIST Model Lab", page_icon="✍️", layout="wide")
st.markdown("""<style>
 .block-container {max-width:1080px;padding-top:4rem;padding-bottom:2rem;}
 h1 {letter-spacing:-.045em;font-weight:750!important;}
 .eyebrow {color:#007f73;font-size:.78rem;font-weight:700;letter-spacing:.14em;}
 .hero-sub {color:#637086;font-size:1.03rem;line-height:1.8;margin:0 0 1.6rem;}
 .prediction {font-size:112px;font-weight:760;line-height:1.15;letter-spacing:-.06em;color:#007f73;}
 .prediction-label {font-size:.8rem;color:#637086;margin-top:8px;}
 .model-tag {display:inline-block;border-radius:5px;background:#e8f4f1;color:#00695c;padding:5px 9px;font-size:.8rem;}
 [data-testid="stSidebar"] {border-right:1px solid #e8ebef;}
 div[data-testid="stVerticalBlockBorderWrapper"] {background:white;border-radius:14px;}
 </style>""", unsafe_allow_html=True)

pad = components.declare_component("mnist_drawing", path=str(ROOT / "drawing_component"))
if "models" not in st.session_state:
    st.session_state.models = {}
    st.session_state.startup_errors = []
    for path in sorted((ROOT / "models").glob("*")):
        if path.suffix.lower() not in {".pth", ".pt"}:
            continue
        try:
            loaded = load_model_bytes(path.read_bytes(), path.name)
            if path.name.startswith("00_D46_CNN"):
                loaded.name = "수업 CNN · 기존 학습"
            st.session_state.models[loaded.identifier] = loaded
        except Exception as error:
            st.session_state.startup_errors.append(f"{path.name}: {error}")

with st.sidebar:
    st.markdown("### MNIST Model Lab")
    st.caption("내가 학습한 모델로 숫자 알아보기")
    st.divider()
    st.markdown("**모델 추가**")
    uploads = st.file_uploader("모델 파일", type=["pth", "pt"], label_visibility="collapsed",
                               accept_multiple_files=True,
                               help="본인이 학습한 state_dict 또는 TorchScript 파일을 선택하세요. 최대 25MB/파일.")
    custom_structure = st.checkbox("Sequential 코드 직접 입력", key="custom_structure")
    sequential_code, input_layout = None, "image"
    if custom_structure:
        st.caption("학습에 사용한 모델 구조를 붙여 넣으세요. 선택한 state_dict 파일에 적용돼요.")
        sequential_code = st.text_area("Sequential 코드", value=EXAMPLE_CODE,
                                       height=280, key="sequential_source")
        input_layout = st.selectbox("모델 입력 형태", ["image", "flat"],
                                   format_func=lambda value: "이미지 [B, 1, 28, 28]" if value == "image" else "펼친 이미지 [B, 784]",
                                   key="sequential_layout")
        with st.expander("코드 입력 도움말"):
            st.write("model = nn.Sequential(...) 또는 nn.Sequential(...)을 입력하세요. 여러 Sequential 블록과 OrderedDict로 이름 붙인 층도 사용할 수 있어요.")
            st.caption("학습 코드의 층 순서·크기·이름을 맞춰 주세요. 입력은 수업과 같은 정규화(mean=0.1307, std=0.3081)를 적용하고, 출력은 숫자 0~9의 점수 10개를 사용해요.")
            st.caption("지원하는 층: " + ", ".join(LAYER_NAMES))
    if st.button("선택한 파일 추가", disabled=not uploads, use_container_width=True):
        added, duplicates, errors = 0, 0, []
        for uploaded in uploads:
            try:
                loaded = load_model_bytes(uploaded.getvalue(), uploaded.name,
                                          sequential_code=sequential_code, input_layout=input_layout)
                if loaded.identifier in st.session_state.models:
                    duplicates += 1
                else:
                    st.session_state.models[loaded.identifier] = loaded
                    added += 1
            except Exception as error:
                errors.append(f"{uploaded.name}: {error}")
        st.session_state.upload_notice = (added, duplicates, errors)
    if "upload_notice" in st.session_state:
        added, duplicates, errors = st.session_state.upload_notice
        if added:
            st.success(f"모델 {added}개를 추가했어요.")
        if duplicates:
            st.caption(f"이미 등록된 파일 {duplicates}개는 유지했어요.")
        for error in errors:
            st.error(error)
    for error in st.session_state.startup_errors:
        st.error(error)
    st.caption("추가한 모델은 현재 접속에서 유지돼요.")
    st.divider()
    registry = st.session_state.models
    selected_id = None
    if registry:
        selected_id = st.selectbox(
            "사용할 모델", options=list(registry),
            format_func=lambda key: f"{registry[key].name} · {registry[key].kind}",
        )
        st.caption(f"등록된 모델 {len(registry)}개")
    else:
        st.info("모델 파일을 추가하면 테스트를 시작할 수 있어요.")

st.markdown('<div class="eyebrow">HANDWRITING EXPERIMENT</div>', unsafe_allow_html=True)
st.title("같은 숫자, 다른 모델")
st.markdown('<p class="hero-sub">숫자를 그리고 모델을 바꿔 보세요. 각 모델이 내 손글씨를 어떻게 읽는지 비교할 수 있어요.</p>', unsafe_allow_html=True)

left, right = st.columns([1.05, 1], gap="large")
prepared = None
with left:
    with st.container(border=True):
        mode = st.radio("입력 방법", ["직접 그리기", "MNIST 예제"], horizontal=True, label_visibility="collapsed")
        if mode == "직접 그리기":
            st.markdown("#### 숫자 하나를 그려 주세요")
            result = pad(initial_image=st.session_state.get("drawing_image"),
                         key="persistent_drawing", default=None)
            if result is not None:
                st.session_state.drawing_image = result.get("image")
            try:
                prepared = prepare_drawing(st.session_state.get("drawing_image"))
            except (ValueError, OSError) as error:
                st.error(f"그림을 읽을 수 없어요: {error}")
        else:
            samples = sorted((ROOT / "samples").glob("*.png"))
            if samples:
                sample = st.selectbox("예제 숫자", samples, format_func=lambda path: path.stem)
                prepared = Image.open(sample).convert("L")
                st.image(prepared.resize((280, 280), Image.Resampling.NEAREST), caption="실제 MNIST 테스트 이미지", width=280)
            else:
                st.info("등록된 예제 이미지가 없어요. 직접 그리기를 사용해 주세요.")
        st.caption("모델을 바꿔도 입력한 숫자는 그대로 유지돼요.")

with right:
    with st.container(border=True):
        st.markdown("#### 예측 결과")
        if selected_id is None:
            st.info("왼쪽 메뉴에서 모델을 추가해 주세요.")
        elif prepared is None:
            st.markdown('<div class="prediction" style="color:#c5cdd6">?</div>', unsafe_allow_html=True)
            st.write("그림을 입력하면 바로 예측해요.")
        else:
            loaded = registry[selected_id]
            try:
                digit, probabilities = predict(loaded, prepared)
                st.markdown(f'<span class="model-tag">{html.escape(loaded.name)} · {loaded.kind}</span>', unsafe_allow_html=True)
                st.markdown('<div class="prediction-label">모델이 읽은 숫자</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prediction">{digit}</div>', unsafe_allow_html=True)
                st.write(f"**선택한 숫자의 확률 {probabilities[digit]:.1%}**")
                probability_frame = pd.DataFrame({"숫자": [str(i) for i in range(10)],
                                                  "확률 (%)": probabilities * 100})
                chart = alt.Chart(probability_frame).mark_bar(color="#007F73").encode(
                    x=alt.X("숫자:N", sort=[str(i) for i in range(10)], axis=alt.Axis(labelAngle=0), title=None),
                    y=alt.Y("확률 (%):Q", scale=alt.Scale(domain=[0, 100]), title="확률 (%)"),
                    tooltip=["숫자:N", alt.Tooltip("확률 (%):Q", format=".2f")],
                ).properties(height=160)
                st.altair_chart(chart, use_container_width=True)
                with st.expander("모델에 전달된 28×28 이미지"):
                    st.image(prepared.resize((168, 168), Image.Resampling.NEAREST), width=168)
                    st.caption("흰 숫자·검은 배경. 손글씨는 크기와 중심을 맞춘 뒤 수업과 같은 정규화를 적용해요.")
            except Exception as error:
                st.error(f"이 모델의 예측을 완료하지 못했어요: {error}")

if selected_id:
    loaded = registry[selected_id]
    with st.expander("선택한 모델의 학습 기록"):
        metadata = loaded.metadata
        if "test" in metadata:
            a, b, c = st.columns(3)
            a.metric("테스트 정확도", f"{metadata['test']['accuracy']:.2%}")
            b.metric("학습한 회차", metadata.get("epochs_ran", "—"))
            c.metric("저장한 회차", metadata.get("best_epoch", "—"))
            if metadata.get("smoke_test"):
                st.caption("소규모 실행 확인용 모델입니다.")
        else:
            st.caption("이 파일에는 학습 결과가 포함되어 있지 않아요.")
        st.caption(f"구조: {metadata.get('architecture', 'TorchScript에 포함')} · 형식: {loaded.kind}")
        if "sequential_code" in metadata:
            st.code(metadata["sequential_code"], language="python")

if prepared is not None and len(registry) > 1:
    with st.expander("같은 숫자로 모든 모델 비교"):
        if st.button("등록된 모델 모두 비교", use_container_width=True):
            rows = []
            for loaded in registry.values():
                try:
                    digit, probabilities = predict(loaded, prepared)
                    rows.append({"모델": loaded.name, "형식": loaded.kind,
                                 "예측": str(digit), "선택 확률": f"{probabilities[digit]:.1%}"})
                except Exception as error:
                    rows.append({"모델": loaded.name, "형식": loaded.kind, "예측": "오류", "선택 확률": str(error)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.caption("MNIST Model Lab · API 키 없이, 등록한 모델로 직접 추론합니다.")
