# 🎬 FULL SCRIPT — CROSS-DOMAIN RECOMMENDATION (NO3 + SINKHORN)

---

## SCENE 1 — PROBLEM: DATA FRAGMENTATION

“Bạn đã bao giờ tự hỏi vì sao đôi khi hệ thống gợi ý không thực sự hiểu bạn? Ví dụ, bạn thường nghe nhạc trên một ứng dụng, xem phim trên một nền tảng khác, và mua sắm ở một nơi hoàn toàn riêng biệt. 

Vấn đề ở đây là mỗi hệ thống chỉ nhìn thấy một phần rất nhỏ trong hành vi của bạn. Trong thực tế, người dùng không tồn tại trong một hệ thống duy nhất mà phân tán trên nhiều nền tảng khác nhau, và mỗi nền tảng giống như một ‘hòn đảo dữ liệu’ tách biệt, không kết nối với nhau. Điều này dẫn đến một vấn đề cốt lõi: dữ liệu bị phân mảnh. 

Khi dữ liệu bị phân mảnh, các hệ thống recommender không thể xây dựng được một bức tranh đầy đủ về sở thích của người dùng. Kết quả là chúng ta gặp phải hai vấn đề rất quen thuộc: data sparsity, khi dữ liệu quá ít để học hiệu quả, và cold-start, khi hệ thống không có đủ thông tin để đưa ra những gợi ý chính xác. 

Chính từ những hạn chế này, một hướng nghiên cứu quan trọng đã được đề xuất, đó là Cross-Domain Recommendation, nhằm tận dụng thông tin từ nhiều domain khác nhau để hiểu người dùng tốt hơn.”

---

## SCENE 2 — WHAT IS CROSS-DOMAIN RECOMMENDATION?

“Cross-Domain Recommendation là cách tiếp cận sử dụng thông tin từ nhiều lĩnh vực khác nhau để nâng cao chất lượng gợi ý. Thay vì chỉ dựa vào một nguồn dữ liệu đơn lẻ, mô hình có thể học thêm từ các domain liên quan, từ đó hiểu người dùng đầy đủ hơn và cải thiện khả năng dự đoán.

Dựa trên mức độ chia sẻ dữ liệu giữa các domain, bài toán này thường được chia thành bốn kịch bản. Đầu tiên là user overlap, khi các domain có chung người dùng nhưng khác item. Thứ hai là item overlap, khi một số item xuất hiện ở nhiều domain dù người dùng không trùng nhau. Thứ ba là trường hợp cả user và item đều overlap, tức là có sự giao nhau rõ ràng ở cả hai phía, giúp việc học trở nên dễ dàng hơn.

Và cuối cùng là trường hợp khó nhất: không có bất kỳ overlap nào. Không có user chung, không có item chung, và cũng không có bất kỳ liên kết rõ ràng nào giữa hai domain. Đây chính là kịch bản thách thức nhất, và cũng là trọng tâm mà paper này tập trung giải quyết.”

---

## SCENE 3 — NO3 SETTING

“Cụ thể hơn, paper này định nghĩa một setting gọi là NO3, viết tắt của No user overlap, No item overlap, và No side information. Điều này có nghĩa là hai domain hoàn toàn tách biệt, không có bất kỳ thông tin chung nào để làm cầu nối. 

Chúng ta không biết người dùng ở domain này có phải là người dùng ở domain kia hay không, cũng không có metadata hay thông tin bổ sung nào để hỗ trợ. Đây là một bài toán cực kỳ khó, bởi vì hầu hết các phương pháp trước đây đều dựa vào một dạng overlap nào đó. 

Tuy nhiên, setting này lại rất gần với thực tế, đặc biệt khi các ràng buộc về quyền riêng tư khiến việc chia sẻ dữ liệu trở nên hạn chế. Do đó, câu hỏi đặt ra là: liệu chúng ta có thể học được mối liên hệ giữa các domain chỉ dựa trên hành vi riêng lẻ của từng domain hay không?”

---

## SCENE 4 — LEARNING OBJECTIVE

“Để tiếp cận bài toán này, mô hình được thiết kế để học đồng thời trên cả hai domain. Mỗi domain có một hàm loss riêng, phản ánh khả năng gợi ý trong domain đó. Sau đó, hai hàm loss này được kết hợp lại thành một objective chung. 

Điều này tương tự như một bài toán multi-task learning, trong đó mô hình học nhiều nhiệm vụ song song. Tuy nhiên, điểm quan trọng không chỉ nằm ở việc tối ưu hai loss riêng biệt, mà là làm sao để hai domain có thể chia sẻ thông tin ở mức biểu diễn. Nói cách khác, chúng ta cần một cơ chế để kết nối hai không gian embedding, dù không có bất kỳ overlap nào ban đầu.”

---

## SCENE 5 — HNO3 (HARD MATCHING)

“Cách tiếp cận đầu tiên là hard matching. Ý tưởng là ghép từng user ở domain này với một user ở domain kia, tạo thành một mapping một-một. Phương pháp này thường sử dụng Hungarian Algorithm để tìm matching tối ưu. 

Sau khi ghép, hai domain được xem như đã có overlap và có thể huấn luyện chung. Tuy nhiên, cách tiếp cận này có một nhược điểm lớn: nó là rời rạc. Việc ghép cặp không thể được tối ưu trực tiếp bằng gradient descent.”

---

## SCENE 6 — LIMITATION

“Do tính chất rời rạc, hard matching gặp nhiều hạn chế. Kết quả phụ thuộc rất nhiều vào chất lượng embedding ban đầu. Nếu embedding chưa tốt, việc ghép cặp sẽ sai lệch. Ngoài ra, vì không differentiable (khả vi), phương pháp này không thể học end-to-end. 

Điều này khiến quá trình huấn luyện kém linh hoạt và khó tối ưu. Đây chính là lý do cần một cách tiếp cận khác mềm hơn và liên tục hơn.”

---

## SCENE 7 — SNO3 (SOFT MATCHING)

“Như chúng ta đã đề cập ở trước, hard matching có một điểm yếu chí mạng là ràng buộc quá lớn giữa các user ở từng domain, khiến mô hình thiếu linh hoạt và khó tối ưu end-to-end.

Soft matching ra đời để giải quyết vấn đề này. Thay vì ép từng user phải match một-một, soft matching cho phép mỗi user liên kết với nhiều user khác với các mức độ khác nhau. Cụ thể, với mỗi user ở domain nguồn, ta gán một vector xác suất (giống như softmax), thể hiện mức độ kết nối tới từng user ở domain đích, sao cho tổng toàn bộ các mức độ liên kết này luôn bằng 1.

Nói cách khác, thay vì một mapping rời rạc, chúng ta có một phân phối xác suất trên toàn bộ các cặp user. Nhờ đó, mô hình linh hoạt hơn, có thể học các mối quan hệ phức tạp giữa hai domain.

Điểm mạnh là quá trình này hoàn toàn khả vi, cho phép tối ưu trực tiếp bằng gradient descent. Soft matching cũng mở đường cho các phương pháp như optimal transport, giúp căn chỉnh hai phân phối một cách hiệu quả.”

---

## SCENE 8 — SINKHORN INTUITION

“Bài toán này được mô hình hóa bằng optimal transport. Thay vì cố gắng tìm một ánh xạ cứng giữa từng user, chúng ta nhìn toàn bộ người dùng trong mỗi domain như một phân phối xác suất, và mục tiêu là biến đổi phân phối này thành phân phối kia với chi phí thấp nhất.

Cụ thể, mỗi user được xem như một ‘khối lượng’. Nhiệm vụ của mô hình là phân bổ khối lượng này từ domain nguồn sang domain đích sao cho tổng chi phí là nhỏ nhất.

Tuy nhiên, việc giải trực tiếp bài toán optimal transport là rất tốn kém. Đây là lúc Sinkhorn algorithm xuất hiện. Thuật toán này thêm một cơ chế làm mượt thông qua regularization, sau đó lặp đi lặp lại việc chuẩn hóa ma trận để nhanh chóng tìm ra một phương án vận chuyển gần tối ưu.

Điểm quan trọng là toàn bộ quá trình này đều gồm các phép toán liên tục như exponential và normalization, nên hoàn toàn differentiable. Điều này cho phép Sinkhorn được tích hợp trực tiếp vào quá trình huấn luyện, giúp mô hình học được cách “match” hai domain một cách mềm dẻo và hiệu quả.”


---

## SCENE 9 — FINAL OBJECTIVE

“Mô hình cuối cùng tối ưu hai thành phần. Thứ nhất là recommendation loss, đảm bảo độ chính xác của gợi ý. 

Thứ hai là Sinkhorn loss, giúp căn chỉnh hai không gian embedding. Hai thành phần này kết hợp với nhau, vừa đảm bảo hiệu năng, vừa đảm bảo sự đồng bộ giữa hai domain.”

---

## SCENE 10 — KEY INSIGHT

“Insight quan trọng nhất của paper là: chúng ta không cần biết người dùng là ai. Không cần identity, không cần thông tin bổ sung. Chỉ cần hiểu họ thích gì. 

Bằng cách căn chỉnh các phân phối sở thích, chúng ta có thể kết nối các domain hoàn toàn tách biệt. Đây chính là sức mạnh của soft matching.”

---

## SCENE 11 — GRADIENT DESCENT

“Quá trình học của mô hình được thực hiện thông qua gradient descent. Tại mỗi bước, mô hình tính gradient của hàm loss, và sử dụng thông tin này để cập nhật tham số theo hướng làm giảm loss. Quá trình này diễn ra lặp đi lặp lại, từng bước nhỏ một, cho đến khi mô hình hội tụ. 

Điều quan trọng ở đây là: cách mà mô hình học phụ thuộc rất nhiều vào hình dạng của hàm loss. Nếu loss không trơn hoặc có nhiều điểm gãy, quá trình tối ưu sẽ trở nên khó khăn. Ngược lại, nếu loss mượt mà và liên tục, mô hình sẽ học hiệu quả hơn.”

---

## SCENE 12 — LOSS LANDSCAPE

“Hãy tưởng tượng hàm loss như một bề mặt trong không gian. Nếu bề mặt này gồ ghề, có nhiều điểm gãy và local minima, thì quá trình tối ưu sẽ dễ bị mắc kẹt. Điều này thường xảy ra với các phương pháp rời rạc. 

Ngược lại, nếu bề mặt loss mượt mà và liên tục, gradient sẽ ổn định hơn, và mô hình có thể tiến dần đến nghiệm tốt hơn. Sự khác biệt này đóng vai trò rất quan trọng khi so sánh giữa hard matching và soft matching trong bài toán này.”

---

## SCENE 13 — 3D EMBEDDING SPACE

“Ở một góc nhìn khác, mỗi người dùng có thể được biểu diễn như một điểm trong không gian embedding. Nếu hình dung trong không gian 3 chiều, ban đầu hai domain sẽ tạo thành hai cụm điểm hoàn toàn tách biệt — giống như hai đám mây nằm xa nhau. Mỗi cụm mang những đặc trưng riêng, phản ánh hành vi của người dùng trong từng domain, và giữa chúng gần như không có bất kỳ liên kết nào.

Mục tiêu của chúng ta là căn chỉnh hai cụm điểm này, để mô hình có thể học được những đặc trưng chung và chuyển giao kiến thức giữa các domain. Tuy nhiên, cách chúng ta thực hiện việc căn chỉnh này là cực kỳ quan trọng.

Một cách đơn giản là sử dụng hard matching — tức là ép từng điểm ở domain này phải khớp trực tiếp với một điểm cụ thể ở domain kia. Nghe thì có vẻ hợp lý, nhưng trên thực tế, cách làm này gây ra một vấn đề lớn. Khi bị ép khớp như vậy, các điểm embedding sẽ bị kéo lại gần nhau một cách cưỡng bức, dẫn đến hiện tượng ‘xâm lấn’ giữa hai cụm dữ liệu. Các điểm bắt đầu chồng lấn lên nhau, ranh giới giữa hai domain bị xóa mờ, và những đặc trưng riêng biệt của từng domain dần biến mất. Kết quả là mô hình học được một biểu diễn bị “trung bình hóa”, không còn phản ánh đúng bản chất của từng nhóm dữ liệu.

Ngược lại, soft matching tiếp cận vấn đề theo một cách linh hoạt hơn. Thay vì ép một điểm phải khớp với duy nhất một điểm khác, mỗi điểm sẽ được phân bổ một cách ‘mềm’ đến nhiều điểm ở domain còn lại, với các mức độ khác nhau. Điều này giúp giữ được cấu trúc tổng thể của từng cụm embedding, đồng thời vẫn tạo ra sự liên kết cần thiết giữa hai domain. Có thể hình dung rằng, thay vì bị kéo thẳng đến một vị trí cụ thể, mỗi điểm sẽ được ‘kéo nhẹ’ về phía cả một cụm điểm tương ứng ở domain kia.

Nhờ vậy, mô hình có thể học được những điểm tương đồng giữa hai domain, mà không làm mất đi sự khác biệt vốn có — điều mà hard matching thường không làm được.”

---

## SCENE 14 — SINKHORN CONVERGENCE

“Để hiểu cách vận hành của thuật toán Sinkhorn, trước hết ta bắt đầu với một ma trận chi phí (cost matrix). Ma trận này phản ánh mức độ khác biệt hoặc tương đồng giữa các đối tượng, tùy theo cách ta định nghĩa.

Ví dụ, một user thích chơi cầu lông có thể được xem là gần với một user thường mua giày thể thao — nếu ta cho rằng hai hành vi này có liên quan. Khi đó, chi phí giữa hai user sẽ thấp, phản ánh mức độ tương đồng cao hơn.

Đây chính là đầu vào quan trọng của bài toán optimal transport.

Sinkhorn algorithm hoạt động bằng cách lặp lại hai bước chuẩn hóa: theo hàng và theo cột. Bắt đầu từ một ma trận vận chuyển ngẫu nhiên, thuật toán liên tục điều chỉnh cho đến khi hội tụ. Kết quả là một kế hoạch vận chuyển ổn định, thể hiện cách phân phối khối lượng giữa hai domain một cách tối ưu.”

---

## SCENE 15 — GRADIENT VECTOR FIELD

“Gradient không chỉ là một vector tại một điểm, mà là một trường vector trên toàn bộ không gian embedding. Mỗi điểm có một hướng di chuyển riêng. Khi training diễn ra, toàn bộ không gian được định hình lại, và các phân phối dần được kéo lại gần nhau.”

---

## SCENE 16 — KL VS SINKHORN

“Để hiểu rõ hơn vì sao paper không sử dụng KL divergence, chúng ta cần nhìn vào bản chất của nó. KL divergence đo sự khác biệt giữa hai phân phối bằng cách so sánh xác suất tại từng điểm tương ứng. Điều này có nghĩa là nó ngầm giả định rằng hai phân phối phải ‘chồng lên nhau’ ở một mức độ nào đó — tức là phải có overlap.

Nếu hai phân phối hoàn toàn tách biệt, ví dụ như trong setting NO3 khi hai domain không có bất kỳ liên kết nào, thì sẽ tồn tại những vùng mà một phân phối có xác suất khác 0, còn phân phối kia lại bằng 0. Khi đó, KL divergence sẽ trở nên không xác định hoặc tiến tới vô cực, dẫn đến việc gradient không còn hữu ích cho quá trình học.

Ngược lại, Sinkhorn distance — dựa trên optimal transport — không yêu cầu hai phân phối phải overlap từ đầu. Thay vì so sánh trực tiếp tại từng điểm, nó tìm cách ‘di chuyển’ khối lượng từ phân phối này sang phân phối kia với chi phí thấp nhất. Nhờ đó, ngay cả khi hai phân phối nằm xa nhau trong không gian embedding, mô hình vẫn có thể nhận được tín hiệu gradient ổn định để học cách căn chỉnh chúng.

Chính sự khác biệt này khiến Sinkhorn trở thành lựa chọn phù hợp hơn trong các bài toán cross-domain khó, nơi mà dữ liệu ban đầu gần như không có sự giao nhau.

Vì vậy, trong các bài toán học biểu diễn hiện đại, việc khai thác cấu trúc hình học của dữ liệu quan trọng không kém việc so sánh xác suất. Và đó chính là lý do các phương pháp dựa trên optimal transport ngày càng trở nên phổ biến.”

---

## SCENE 17 — WASSERSTEIN CONNECTION

“Ý tưởng này có mối liên hệ chặt chẽ với Wasserstein distance, một khái niệm đã được sử dụng rất hiệu quả trong GAN. Trong các mô hình GAN truyền thống, khi hai phân phối của dữ liệu thật và dữ liệu sinh ra không overlap, các độ đo như KL divergence hay Jensen-Shannon divergence thường dẫn đến gradient rất yếu hoặc không ổn định, khiến việc huấn luyện trở nên khó khăn.

Wasserstein distance giải quyết vấn đề này bằng cách đo ‘khoảng cách’ giữa hai phân phối dựa trên chi phí vận chuyển khối lượng — tương tự như bài toán optimal transport. Nhờ đó, ngay cả khi hai phân phối hoàn toàn tách biệt, nó vẫn cung cấp một tín hiệu gradient có ý nghĩa, giúp mô hình tiếp tục học.

## KẾT

“Chính vì vậy, trong bối cảnh cross-domain recommendation, Sinkhorn không chỉ là một công cụ kỹ thuật, mà còn là chìa khóa giúp mô hình học được cách kết nối hai domain vốn dĩ hoàn toàn tách biệt.

“Và đây cũng chính là nội dung cuối cùng của phần trình bày. Xin chân thành cảm ơn thầy và các bạn đã lắng nghe.”
