# Vietnamese Syllable-level Bigram Language Model

## Đề bài

**[BT 01] Xây dựng mô hình ngôn ngữ n-gram âm tiết cho tiếng Việt**

**Yêu cầu:**
1. Xây dựng một Mô hình ngôn ngữ [bigram] cho tiếng Việt mức âm tiết.
2. Tính xác suất của một câu "Hôm nay trời đẹp lắm"
3. Sinh ra một số câu bằng mô hình đã xây dựng

**Gợi ý:**
- Hãy tải về một corpus tiếng Việt
- Hãy sử dụng `<s>` và `</s>`

## Dataset

Vietnamese Wikipedia từ HuggingFace Datasets:
- **Tên dataset**: `wikimedia/wikipedia` (subset `20231101.vi`)
- **Link**: https://huggingface.co/datasets/wikimedia/wikipedia
- **Số bài viết sử dụng**: 5,000 bài
- **Tổng dung lượng**: ~50 triệu ký tự

## Cài đặt

```bash
pip install datasets
```

## Chạy chương trình

```bash
python bigram_model.py
```

Có thể thay đổi số bài viết trong `main()` bằng cách sửa `num_articles`:
```python
corpus = load_corpus(num_articles=5000)  # Mặc định 5000 bài
```

## Cấu trúc chương trình

| Phần | Mô tả |
|------|--------|
| `load_corpus()` | Tải dữ liệu Vietnamese Wikipedia từ HuggingFace |
| `preprocess()` | Lowercase, tách câu, tokenize thành âm tiết, thêm `<s>` / `</s>` |
| `build_bigram_model()` | Đếm tần suất bigram và unigram |
| `bigram_probability()` | Tính P(w2\|w1) với MLE hoặc Laplace smoothing |
| `sentence_probability()` | Tính xác suất chung của một câu |
| `generate_sentence()` | Sinh câu bằng cách lấy mẫu từ phân phối bigram |

## Kết quả mẫu

### Thống kê corpus

```
Đã tải 5000 bài viết, tổng 50,868,835 ký tự.
Số câu sau tiền xử lý: 534,837
Số bigram khác nhau: 1,587,615
Số unigram khác nhau: 131,138
```

### Top 20 Bigram (theo tần suất)

```
  P(nam          | việt        ) = 24676/34113 = 0.7234
  P(thể          | có          ) = 22561/110993 = 0.2033
  P(một          | là          ) = 19134/143013 = 0.1338
  P(các          | <s>         ) = 18209/534837 = 0.0340
  P(trong        | <s>         ) = 14768/534837 = 0.0276
  P(dụng         | sử          ) = 13842/24419 = 0.5669
  P(số           | một         ) = 12077/106466 = 0.1134
  P(gia          | quốc        ) = 10976/46320 = 0.2370
  P(phố          | thành       ) = 10855/55550 = 0.1954
  P(là           | gọi         ) =  9894/15791 = 0.6266
  P(giới         | thế         ) =  9612/29464 = 0.3262
  P(triển        | phát        ) =  9416/21858 = 0.4308
  P(tiên         | đầu         ) =  9178/32033 = 0.2865
  P(năm          | <s>         ) =  9167/534837 = 0.0171
  P(các          | của         ) =  8998/162602 = 0.0553
  P(quốc         | trung       ) =  8822/32081 = 0.2750
  P(một          | <s>         ) =  8770/534837 = 0.0164
  P(năm          | vào         ) =  8448/44602 = 0.1894
  P(</s>         | nam         ) =  7740/47021 = 0.1646
  P(chức         | tổ          ) =  7472/10933 = 0.6834
```

### Xác suất câu "Hôm nay trời đẹp lắm"

**MLE (không smoothing):**

```
  P(hôm        | <s>       ) = 0.000118
  P(nay        | hôm       ) = 0.212991
  P(trời       | nay       ) = 0.000196
  P(đẹp        | trời      ) = 0.000389
  P(lắm        | đẹp       ) = 0.000000    ← bigram chưa xuất hiện trong corpus
  P(</s>       | lắm       ) = 0.246575

  P("Hôm nay trời đẹp lắm") = 0.0000000000e+00
  Log2 P = -inf
```

> P = 0 vì bigram ("đẹp", "lắm") không xuất hiện trong corpus Wikipedia.

**Laplace smoothing (V=131,138):**

```
  P(hôm        | <s>       ) = 0.000096
  P(nay        | hôm       ) = 0.001077
  P(trời       | nay       ) = 0.000021
  P(đẹp        | trời      ) = 0.000022
  P(lắm        | đẹp       ) = 0.000008
  P(</s>       | lắm       ) = 0.000419

  P("Hôm nay trời đẹp lắm") = 1.5294070909e-25
  Log2 P = -82.4352
```

### So sánh xác suất hai câu (Laplace smoothing)

```
  "Hôm nay trời đẹp lắm":  P = 1.529407e-25,  Log2 P = -82.4352
  "Việt Nam là một quốc gia":  P = 7.672697e-14,  Log2 P = -43.5673
```

> Câu "Việt Nam là một quốc gia" có xác suất cao hơn nhiều vì các bigram đều phổ biến trong Wikipedia.

### Câu sinh ra từ mô hình bigram

```
  [1] pháp
  [2] ảnh hưởng nghiêm là trứng kiến thức của một loại chính sách chuyển động vật chính yếu tố giống các
  [3] tên gọi thổ được sự kết có trách chức quản lý câu cá thể đồng mỹ sơn gồm viện đại
  [4] những cải tiến hành chính thức đi đánh chiếm được coi giao việt nam và bách khoa học cro3
  [5] mô phỏng sinh chính thức của mình trở thành thuật mạnh mẽ và kết n trong các công nhân dân
```

> Các câu sinh ra có cấu trúc cục bộ hợp lý (các cặp từ liền kề có nghĩa) nhưng thiếu ngữ nghĩa toàn cục — đây là hạn chế cố hữu của mô hình bigram.
