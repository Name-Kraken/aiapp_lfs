import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertJapaneseTokenizer

# warning を防ぐコード
import warnings
warnings.filterwarnings('ignore')

# ラベルの名前のリストを定義します。
label_names = ["1.安らぐ本能", "2.進める本能", "3.決する本能", "4.有する本能", "5.属する本能", "6.高める本能", "7.伝える本能", "8.物語る本能"]

# 分かち書き用の tokenizer
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')


# モデルの定義
class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        self.fc = nn.Linear(768, 8)

    def forward(self, x):
        bert_out = self.bert(x, output_attentions=True)
        h = bert_out[0][:,0,:]
        h = self.fc(h)
        return h, bert_out[2]

def predict(text):
    # モデルのロード
    net = BertClassifier()
    net.load_state_dict(torch.load('model.pth'))

    # テキストをトークン化
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # 512トークンに制限
    if input_ids.size(1) > 512:
        st.warning("入力テキストが512トークンを超えています。最初の512トークンだけが使用されます。")
        input_ids = input_ids[:, :512]

    # GPUが利用可能ならGPUにデータを送る
    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')
        net.to('cuda')

    # モデルの推論モードをオンにして推論を実行
    net.eval()
    with torch.no_grad():
        outputs = net(input_ids)[0]

    # 各ラベルのスコアを取得し、小数点第2位までに丸める
    scores = [round(float(score), 2) for score in F.softmax(outputs, dim=1)[0]]

    # 最も確率の高いラベルのインデックスを取得
    _, predicted = torch.max(outputs, 1)

    # ラベルのインデックスをPythonのint型に変換
    predicted_label_index = predicted.item()

    # ラベルの名前を取得
    predicted_label_name = label_names[predicted_label_index]

    # 各ラベルのスコアと名前を辞書に格納
    label_scores = {label_names[i]: score for i, score in enumerate(scores)}

    return predicted_label_name, label_scores

import streamlit as st
import pandas as pd

# StreamlitアプリのUI部分

st.title('どの本能活性化されている？')

# ライブラリ追加
from PIL import Image

img = Image.open('logo.jpg')

# use_column_width 実際のレイアウトの横幅に合わせる
st.image(img, caption='', use_column_width=True)

st.text('参考文献')
st.text('著：鈴木 祐')
link = "https://amzn.to/3OC7Q0X"
st.markdown(f"[ヒトが持つ8つの本能に刺さる 進化論マーケティング]({link})")

st.text('テキストによって何の本能が活性化されているのか調べることが出来ます')
st.text('各本能については本を参考にしてください')
st.text('※入力文字数は512token(単語)までです,エラーが出た場合は文字数を削ってください')

text = st.text_area("テキストを入力してください:", value='', max_chars=None, key=None)

if st.button('予測'):
    predicted_label, label_scores = predict(text)
    st.write(f"最も活性化されている本能: {predicted_label}")
    st.write("各ラベルのスコア:")
    for label, score in label_scores.items():
        st.write(f"{label}: {score}")

    # ラベルのスコアをPandas DataFrameに変換
    scores_df = pd.DataFrame(list(label_scores.items()), columns=['Label', 'Score'])
    # ラベルの名前でソート（昇順）
    scores_df = scores_df.sort_values(by='Label')
    # インデックスをラベル名に設定
    scores_df = scores_df.set_index('Label')
    # 棒グラフで表示
    st.bar_chart(scores_df)

# warning を防ぐコード
import warnings
warnings.filterwarnings('ignore')