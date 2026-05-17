# vdxpy

**[VDXP](https://github.com/blackocean-tech/vdxp)（Visual Data eXchange Protocol）の Python 実装 — HDMI 経由の単方向ファイル転送**

[![Throughput](https://img.shields.io/badge/max_throughput-15.1_MB%2Fs-00ff88?style=flat-square)]()
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![SHA256](https://img.shields.io/badge/integrity-SHA256_verified-brightgreen?style=flat-square)]()

vdxpy はファイルを画面上の視覚的セルパターンとして符号化し、HDMI + USB キャプチャデバイスでキャプチャして受信側でデコードします。ネットワーク接続不要。USB ドライブ不要。すべての転送は SHA256 検証済み。

![VDXP 概要](vdxp.png)

```
[送信PC] → 画面にセルパターン表示 → HDMI 出力 →
  → [USB キャプチャデバイス] → USB →
  [受信PC] → デコード → ファイル復元 (SHA256 一致 ✓)
```

---

## クイックスタート（~100 KB/s）

### 要件

- Python 3.10+
- USB HDMI キャプチャデバイス（動作確認済み: [USB-CVHDUVC2](https://www.sanwa.co.jp/product/syohin?code=USB-CVHDUVC2)、~¥15,000）
- PC 2台（または 1台で HDMI ループバック）

### インストール

```bash
pip install -r requirements.txt
```

### ファイル送信

```bash
# 送信PC（画面にパターンを表示）
python src/sender.py --file secret.pdf --profile cvhduvc2

# 受信PC（USB デバイスでキャプチャ）
python src/receiver.py --profile cvhduvc2 --output received.pdf
```

### 検証

```bash
sha256sum secret.pdf received.pdf
# 両方のハッシュが一致 ✓
```

---

## 仕組み

vdxpy はデータを画面上の視覚的セルパターンとして符号化します:

1. **送信側** がファイルをチャンクに分割し、Reed-Solomon 誤り訂正を適用、各チャンクを色/グレースケールセルのグリッドとして HDMI 出力に描画
2. **キャプチャデバイス** が HDMI 信号を USB 経由でデジタル化
3. **受信側** がセル値をデコード、RS で誤り訂正、ファイルを再構成

### 主要技術

- **BGR 立方体頂点 8色パレット** — クロマ距離を最大化し MJPEG 圧縮アーティファクトに耐える
- **Y-only グレースケール (Y16/Y32/Y64)** — クロマサブサンプリング干渉を完全に排除
- **フィンガープリントスキップ** — 32セルハッシュで重複フレームを ~0.1ms で識別
- **3段階デコードフィルタ** — フィンガープリント → ヘッダのみ → フルデコード、CPU 負荷を最小化
- **プロファイルシステム** — デバイスと環境ごとに最適化されたプロファイル

---

## 性能

| 構成 | スループット | ハードウェア |
|------|------------|------------|
| Free (MJPEG, 8色, cs8) | ~100 KB/s | CVHDUVC2 または互換 USB2.0 MJPEG キャプチャ |
| Accelerated Engine 使用時 (YUY2, Y32, cs1) | ~15,000 KB/s | USB3.0 YUY2 デバイス + Accel Engine |

すべてのベンチマークはランダムデータペイロード (5 MB -- 500 MB) で SHA256 検証済み。

---

## 独自プロファイルの構築

異なるハードウェアをお持ちの場合、診断ツールで最適化できます:

```bash
# デバイスの色マージンと ECC 使用率を測定
python tools/diagnose.py --profile cvhduvc2 --duration 30

# 1フレームをキャプチャして視覚的に確認
python tools/snap_frame.py --profile cvhduvc2

# キャプチャデバイスの実際の色分布を調査
python tools/probe_colors.py --profile cvhduvc2
```

詳細はチューニングガイド（Accelerated Engine に付属）を参照してください。

---

## 代替手段との比較

| ソリューション | スループット | コスト | タイプ |
|--------------|------------|------|--------|
| **vdxpy + Accel** | **15.1 MB/s** | お問い合わせ | ソフトウェア + USB キャプチャ |
| **vdxpy (Free)** | **100 KB/s** | **無料 + デバイス代** | OSS + USB キャプチャ |
| libcimbar | 106 KB/s | 無料 | カメラベース OSS |
| TGXf | ~4 KB/s | 無料 | QR コードストリーム |
| 光ファイバデータダイオード | 1--100 Gbps | $5,000--$100,000+ | 専用ハードウェア |

---

## ユースケース

- **防衛・官公庁** — 隔離されたネットワークセグメント間のデータ転送
- **産業制御 (OT/ICS)** — エアギャップネットワークからの SCADA データエクスポート
- **金融** — 規制ネットワーク境界を越えたデータ移動

---

## Accelerated Engine

オプションの Accelerated Engine により、USB3.0 YUY2 ハードウェアとの組み合わせで最大約 150 倍のスループット向上を実現:

- カスタム Cython Reed-Solomon デコーダ（Berlekamp-Massey + Chien + Forney）
- OpenMP 並列ブロックデコード（標準 RS 比最大 90.5 倍高速）
- ゼロコピーメモリアクセスによる最適化セルサンプリング

Accelerated Engine は動作確認済み USB キャプチャデバイスとのセットで国内向けに有償提供可能です。
詳細はお問い合わせください: **contact@blackocean.tech**

---

## フィードバック

バグ報告や機能リクエストは [Issues](https://github.com/blackocean-tech/vdxpy/issues) で受け付けています。現時点ではプルリクエストのレビューは行っていません。

---

## ライセンス

MIT License. 詳細は [LICENSE](LICENSE) を参照。

---

*すべてのベンチマークは SHA256 検証済みのランダムデータで実施。出力ハッシュが入力ハッシュと完全に一致しない限り、転送成功とは報告されません。*
