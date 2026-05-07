# Lab 21 - Evaluation Report

**Học viên**: Nguyễn Thị Thùy Trang - 2A202600214  
**Ngày nộp**: 07/05/2026  
**Notebook**: `notebooks/Lab21_LoRA_Finetuning_T4.ipynb`  
**Submission option**: A - Lightweight ZIP

## 1. Setup

- **Base model**: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`
- **Số lượng mẫu**: 200 samples, chia 90% train và 10% eval
- **Định dạng dữ liệu**: Alpaca format gồm `instruction`, `input`, `output`, sau đó format thành cột `text`
- **GPU**: T4 16GB trên Google Colab
- **Kỹ thuật fine-tuning**: QLoRA 4-bit + LoRA adapter + Unsloth + TRL `SFTTrainer`
- **Cấu hình train chung**:
  - `num_train_epochs = 3`
  - `learning_rate = 2e-4`
  - `lr_scheduler_type = "cosine"`
  - `per_device_train_batch_size = 1`
  - `gradient_accumulation_steps = 8`
  - `optim = "adamw_8bit"`
  - `target_modules = ["q_proj", "v_proj"]`

Tổng thời gian train cho 3 rank là khoảng **12.76 phút**. Nếu tính theo mức tham chiếu T4 là **0.35 USD/giờ**, chi phí ước tính khoảng **0.07 USD**.

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|------------------|------------|-----------|-----------|------------|
| 8 | 16 | 2,293,760 | 4.06 min | 11.76 GB | 1.6680 | 5.3015 |
| 16 | 32 | 4,587,520 | 4.44 min | 11.01 GB | 1.6534 | 5.2247 |
| 64 | 128 | 18,350,080 | 4.26 min | 12.75 GB | 1.6548 | 5.2319 |

Kết quả cho thấy rank **r=16** đạt perplexity thấp nhất trong 3 cấu hình. Rank **r=8** có số tham số trainable ít nhất, nhưng perplexity cao hơn. Rank **r=64** có số tham số trainable lớn gấp 4 lần r=16, dùng VRAM cao nhất, nhưng không cải thiện perplexity so với r=16.

## 3. Loss Curve Analysis

Trong cấu hình T4, notebook tắt evaluation trong quá trình training để tránh lỗi OOM, vì vậy loss curve chủ yếu dựa trên training loss. Sau khi train xong, notebook mới chạy evaluation riêng bằng hàm `safe_evaluate()`.

Từ bảng kết quả, eval loss của ba rank không chênh lệch quá lớn:

- r=8: 1.6680
- r=16: 1.6534
- r=64: 1.6548

Không có dấu hiệu r=64 mang lại lợi ích rõ ràng. Việc tăng rank từ 16 lên 64 làm số trainable params tăng mạnh và VRAM tăng lên 12.75 GB, nhưng eval loss lại không giảm. Điều này cho thấy hiện tượng diminishing returns: model có thêm khả năng học, nhưng dataset 200 samples có thể chưa đủ lớn hoặc chưa đủ đa dạng để tận dụng rank cao.

## 4. Qualitative Comparison

### Example 1

**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.

**Base**: Machine learning là một phương pháp xử lý dữ liệu tự động, cho phép máy tính học từ dữ liệu và đưa ra kết luận hoặc dự đoán dựa trên dữ liệu mà nó đã được đào tạo.

**Fine-tuned (r=16)**: Machine learning là một phần của công nghệ nhân tạo, cho phép máy tính học hỏi và cải thiện khả năng phân tích dữ liệu và đưa ra quyết định dựa trên dữ liệu mà nó đã được đào tạo.

**Nhận xét**: Cả hai câu trả lời đều đúng hướng. Fine-tuned trả lời tự nhiên hơn một chút, nhưng vẫn còn một số cụm từ chưa thật chuẩn như "công nghệ nhân tạo".

### Example 2

**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.

**Base**: The

**Fine-tuned (r=16)**: Đ

**Nhận xét**: Cả base và fine-tuned đều thất bại ở prompt này. Đây là một case loss quan trọng, cho thấy model sau fine-tuning chưa ổn định với yêu cầu sinh code Python.

### Example 3

**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.

**Base**: 1. Đảm bảo giao diện thân thiện với người dùng: Thiết kế giao diện thân thiện với người dùng giúp người dùng dễ dàng tiếp cận và sử dụng sản phẩm...

**Fine-tuned (r=16)**: 1. Đảm bảo trải nghiệm người dùng: Thiết kế cho người dùng. Điều này có nghĩa là hãy đặt chính họ ở trung tâm và thiết kế trải nghiệm của họ...

**Nhận xét**: Fine-tuned có câu trả lời gần với yêu cầu hơn và liệt kê theo phong cách ngắn gọn hơn. Tuy nhiên output vẫn bị cắt, nên cần tăng `max_new_tokens` hoặc cải thiện prompt khi evaluate.

### Example 4

**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.

**Base**: LoRA và QLoRA đều là các thuật toán được sử dụng trong công nghệ học máy để giải mã các bản ghi nhị số...

**Fine-tuned (r=16)**: LoRA và QLoRA là hai phương pháp được sử dụng để chuyển đổi dữ liệu không tương tác thành dữ liệu tương tác...

**Nhận xét**: Cả hai câu trả lời đều chưa đúng bản chất. LoRA là kỹ thuật thêm low-rank adapters để fine-tune hiệu quả, còn QLoRA kết hợp quantization 4-bit với LoRA. Đây là ví dụ cho thấy fine-tuning với dataset nhỏ không đảm bảo model hiểu đúng kiến thức chuyên sâu.

### Example 5

**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.

**Base**: Prompt engineering là quá trình tạo ra những câu hỏi hoặc yêu cầu cụ thể để kích hoạt mô hình ngôn ngữ AI trả lời chính xác...

**Fine-tuned (r=16)**: Prompt engineering là một kỹ thuật được sử dụng để tạo ra các prompt có thể tối ưu hóa hiệu suất của AI model trong việc trả lời câu hỏi...

**Nhận xét**: Fine-tuned trả lời đúng một phần về prompt engineering, nhưng chưa phân biệt đầy đủ RAG và fine-tuning trong phần output được ghi lại. Cần kiểm tra full generation để đánh giá công bằng hơn.

## 5. Conclusion về Rank Trade-off

Trong bài, rank **r=16** là lựa chọn tốt nhất về mặt trade-off. So với r=8, r=16 có số trainable params gấp đôi, nhưng đổi lại eval loss và perplexity tốt hơn. Cụ thể, perplexity giảm từ 5.3015 xuống 5.2247, cho thấy adapter có thêm năng lực biểu diễn và học tốt hơn trên dataset tiếng Việt dạng Alpaca format. Tuy nhiên, khi tăng lên r=64, số tham số trainable tăng lên 18,350,080, lớn gấp 4 lần r=16, và peak VRAM tăng lên 12.75 GB. Mặc dù vậy, perplexity của r=64 là 5.2319, không tốt hơn r=16. Điều này cho thấy việc tăng rank không phải lúc nào cũng giúp cải thiện chất lượng, đặc biệt khi dataset chỉ có 200 samples. Với dataset nhỏ, rank quá cao có thể làm tăng chi phí tính toán mà không đem lại lợi ích rõ ràng. Nếu deploy hoặc tiếp tục phát triển bài lab này, em sẽ chọn **r=16** vì đây là cấu hình cân bằng giữa chất lượng, VRAM, thời gian train và số tham số cần lưu.

## 6. What I Learned

- LoRA rank ảnh hưởng trực tiếp đến số tham số trainable, nhưng rank cao hơn không đảm bảo perplexity tốt hơn.
- QLoRA 4-bit giúp fine-tune model lớn hơn trên GPU T4 16GB, nhưng vẫn cần quản lý VRAM bằng batch size nhỏ và gradient accumulation.
- Evaluation cần kết hợp cả metric định lượng và ví dụ qualitative, vì perplexity tốt hơn không có nghĩa là mọi câu trả lời đều tốt hơn trong thực tế.
