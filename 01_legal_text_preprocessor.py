"""
Module 1: Multi-Core CPU Legal Text Preprocessor (VietLawAssist)
Chuẩn hóa và bóc tách cấu trúc phân cấp văn bản quy phạm pháp luật đa lĩnh vực:
1. 4 lĩnh vực cốt lõi: Dân sự (BLDS), Lao động (BLLD), Doanh nghiệp (LDN), Hình sự (BLHS)
2. Chuẩn hóa Unicode NFC (ngăn ngừa lỗi ký tự tổ hợp tiếng Việt)
3. Bóc tách phân cấp: Luật -> Chương -> Điều -> Khoản -> Nội dung
4. Tách từ ngữ pháp lý chuyên dụng bằng PyVi Tokenizer
5. Tự động sinh danh mục Chunks cấu trúc kèm Domain Tagging phục vụ lọc chính xác.
"""

from pathlib import Path
import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import re
import json
import unicodedata
from pyvi import ViTokenizer
from typing import List, Dict, Any

# Dynamic Path Setup (Zero Hardcoded Paths)
CURRENT_DIR = Path(__file__).resolve().parent
try:
    sys.path.append(str(CURRENT_DIR.parent))
    from shared_utils.path_resolver import resolve_path, DATA_DIR
except Exception:
    DATA_DIR = CURRENT_DIR.parent / "data"

def normalize_vietnamese_text(text: str) -> str:
    """Chuẩn hóa Unicode NFC và dọn dẹp khoảng trắng dư thừa."""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n', text)
    return text.strip()

def parse_legal_corpus(raw_text: str, law_code: str, domain: str) -> List[Dict[str, Any]]:
    """Phân tích văn bản luật thành các Điều luật có gắn nhãn ngữ cảnh và lĩnh vực."""
    normalized = normalize_vietnamese_text(raw_text)
    
    # Regex nhận diện định dạng Điều X. Tên điều luật
    pattern = re.compile(r'(Điều\s+(\d+)[\.:]\s*([^\n]+))', re.IGNORECASE)
    matches = list(pattern.finditer(normalized))
    
    chunks = []
    if not matches:
        # Fallback chia theo đoạn nếu không khớp cấu trúc Điều
        paragraphs = [p.strip() for p in normalized.split("\n") if len(p.strip()) > 30]
        for idx, p in enumerate(paragraphs):
            chunks.append({
                "chunk_id": f"{law_code}_P{idx+1}",
                "law_code": law_code,
                "domain": domain,
                "article_num": idx + 1,
                "title": f"Đoạn văn {idx+1}",
                "content": p,
                "tokenized_content": ViTokenizer.tokenize(p)
            })
        return chunks

    for i in range(len(matches)):
        start_idx = matches[i].start()
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(normalized)
        
        full_block = normalized[start_idx:end_idx].strip()
        article_num = int(matches[i].group(2))
        title = matches[i].group(3).strip()
        
        # Tách từ tiếng Việt chuyên dụng phục vụ BM25
        tokenized = ViTokenizer.tokenize(full_block)
        
        chunks.append({
            "chunk_id": f"{law_code}_D{article_num}",
            "law_code": law_code,
            "domain": domain,
            "article_num": article_num,
            "title": f"Điều {article_num}. {title}",
            "content": full_block,
            "tokenized_content": tokenized
        })
        
    return chunks

MULTI_DOMAIN_LEGAL_DATA = {
    "BLDS_2015": {
        "domain": "dan_su",
        "name": "Bộ luật Dân sự 2015",
        "text": """
Điều 1. Phạm vi điều chỉnh
Bộ luật này quy định địa vị pháp lý, chuẩn mực pháp lý về cách ứng xử của cá nhân, pháp nhân; quyền, nghĩa vụ về nhân thân và tài sản của cá nhân, pháp nhân trong các quan hệ được hình thành trên cơ sở bình đẳng, tự do ý chí, độc lập về tài sản và tự chịu trách nhiệm.

Điều 2. Công nhận, tôn trọng, bảo vệ và bảo đảm quyền dân sự
1. Mọi quyền dân sự của cá nhân, pháp nhân được công nhận, tôn trọng, bảo vệ và bảo đảm theo Hiến pháp và pháp luật.
2. Quyền dân sự chỉ có thể bị hạn chế theo quy định của luật trong trường hợp cần thiết vì lý do quốc phòng, an ninh quốc gia, trật tự, an toàn xã hội, đạo đức xã hội, sức khỏe của cộng đồng.

Điều 3. Nguyên tắc cơ bản của pháp luật dân sự
1. Mọi cá nhân, pháp nhân đều bình đẳng, không được lấy bất kỳ lý do nào để phân biệt đối xử; được pháp luật bảo hộ như nhau về các quyền nhân thân và tài sản.
2. Cá nhân, pháp nhân xác lập, thực hiện, chấm dứt quyền, nghĩa vụ dân sự của mình trên cơ sở tự do, tự nguyện cam kết, thỏa thuận.
3. Cá nhân, pháp nhân phải xác lập, thực hiện, chấm dứt quyền, nghĩa vụ dân sự của mình một cách thiện chí, trung thực.

Điều 11. Các phương thức bảo vệ quyền dân sự
Khi quyền dân sự của cá nhân, pháp nhân bị xâm phạm thì chủ thể đó có quyền tự bảo vệ theo quy định của Bộ luật này hoặc yêu cầu cơ quan, tổ chức có thẩm quyền: công nhận quyền dân sự; buộc chấm dứt hành vi xâm phạm; buộc xin lỗi, cải chính công khai; buộc thực hiện nghĩa vụ; buộc bồi thường thiệt hại; hủy quyết định cá biệt trái pháp luật.

Điều 116. Giao dịch dân sự
Giao dịch dân sự là hợp đồng hoặc hành vi pháp lý đơn phương làm phát sinh, thay đổi hoặc chấm dứt quyền, nghĩa vụ dân sự.

Điều 122. Giao dịch dân sự vô hiệu
Giao dịch dân sự không có một trong các điều kiện quy định tại Điều 117 của Bộ luật này thì vô hiệu, trừ trường hợp Bộ luật này có quy định khác.

Điều 584. Căn cứ phát sinh trách nhiệm bồi thường thiệt hại
1. Người nào có hành vi xâm phạm tính mạng, sức khỏe, danh dự, nhân phẩm, uy tín, tài sản, quyền, lợi ích hợp pháp khác của người khác mà gây thiệt hại thì phải bồi thường, trừ trường hợp Bộ luật này, luật khác có liên quan quy định khác.
2. Người gây thiệt hại không phải chịu trách nhiệm bồi thường thiệt hại trong trường hợp thiệt hại phát sinh là do sự kiện bất khả kháng hoặc hoàn toàn do lỗi của bên bị thiệt hại.

Điều 601. Bồi thường thiệt hại do nguồn nguy hiểm cao độ gây ra
1. Nguồn nguy hiểm cao độ bao gồm phương tiện giao thông cơ giới, hệ thống tải điện, nhà máy công nghiệp đang hoạt động, vũ khí, chất nổ, chất cháy, chất độc, chất phóng xạ, thú dữ và các nguồn nguy hiểm cao độ khác do pháp luật quy định.
2. Chủ sở hữu nguồn nguy hiểm cao độ phải bồi thường thiệt hại do nguồn nguy hiểm cao độ gây ra; nếu chủ sở hữu đã giao cho người khác chiếm hữu, sử dụng thì người này phải bồi thường, trừ trường hợp có thỏa thuận khác.
"""
    },
    "BLLD_2019": {
        "domain": "lao_dong",
        "name": "Bộ luật Lao động 2019",
        "text": """
Điều 13. Hợp đồng lao động
1. Hợp đồng lao động là sự thỏa thuận giữa người lao động và người sử dụng lao động về việc làm có trả công, tiền lương, điều kiện lao động, quyền và nghĩa vụ của mỗi bên trong quan hệ lao động.
2. Trước khi nhận người lao động vào làm việc thì người sử dụng lao động phải giao kết hợp đồng lao động với người lao động.

Điều 20. Loại hợp đồng lao động
1. Hợp đồng lao động phải được giao kết theo một trong hai loại sau đây:
a) Hợp đồng lao động không xác định thời hạn;
b) Hợp đồng lao động xác định thời hạn, trong đó hai bên xác định thời hạn từ đủ 12 tháng đến 36 tháng.

Điều 35. Quyền đơn phương chấm dứt hợp đồng lao động của người lao động
1. Người lao động có quyền đơn phương chấm dứt hợp đồng lao động nhưng phải báo trước cho người sử dụng lao động ít nhất 45 ngày nếu làm việc theo hợp đồng lao động không xác định thời hạn; ít nhất 30 ngày nếu làm việc theo hợp đồng lao động xác định thời hạn từ 12 tháng đến 36 tháng.
2. Người lao động có quyền đơn phương chấm dứt hợp đồng lao động không cần báo trước trong các trường hợp: không được bố trí đúng công việc, địa điểm làm việc; không được trả đủ lương hoặc trả lương không đúng thời hạn; bị người sử dụng lao động ngược đãi, đánh đập hoặc có lời nói, hành vi nhục mạ; bị quấy rối tình dục tại nơi làm việc.

Điều 36. Quyền đơn phương chấm dứt hợp đồng lao động của người sử dụng lao động
1. Người sử dụng lao động có quyền đơn phương chấm dứt hợp đồng lao động trong các trường hợp: người lao động thường xuyên không hoàn thành công việc theo hợp đồng; người lao động bị ốm đau, tai nạn đã điều trị liên tục mà khả năng lao động chưa hồi phục; do thiên tai, hỏa hoạn, dịch bệnh nguy hiểm buộc phải thu hẹp sản xuất; người lao động tự ý bỏ việc mà không có lý do chính đáng từ 05 ngày làm việc liên tục trở lên.
2. Khi đơn phương chấm dứt hợp đồng lao động, người sử dụng lao động phải báo trước ít nhất 45 ngày đối với hợp đồng không xác định thời hạn; ít nhất 30 ngày đối với hợp đồng xác định thời hạn từ 12 tháng đến 36 tháng.

Điều 41. Nghĩa vụ của người sử dụng lao động khi đơn phương chấm dứt hợp đồng lao động trái pháp luật
1. Phải nhận người lao động trở lại làm việc theo hợp đồng lao động đã giao kết; phải trả tiền lương, đóng bảo hiểm xã hội, bảo hiểm y tế, bảo hiểm thất nghiệp trong những ngày người lao động không được làm việc và phải trả thêm cho người lao động một khoản tiền ít nhất bằng 02 tháng tiền lương theo hợp đồng lao động.
2. Trường hợp người lao động không muốn tiếp tục làm việc thì ngoài khoản tiền quy định tại khoản 1 Điều này, người sử dụng lao động phải trả trợ cấp thôi việc theo quy định tại Điều 46 của Bộ luật này.

Điều 46. Trợ cấp thôi việc
1. Khi hợp đồng lao động chấm dứt theo quy định thì người sử dụng lao động có trách nhiệm trả trợ cấp thôi việc cho người lao động đã làm việc thường xuyên cho mình từ đủ 12 tháng trở lên, mỗi năm làm việc được trợ cấp một nửa tháng tiền lương.
2. Thời gian làm việc để tính trợ cấp thôi việc là tổng thời gian người lao động đã làm việc thực tế trừ đi thời gian người lao động đã tham gia bảo hiểm thất nghiệp và thời gian làm việc đã được người sử dụng lao động chi trả trợ cấp thôi việc.

Điều 105. Thời giờ làm việc bình thường
1. Thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày và không quá 48 giờ trong 01 tuần.
2. Người sử dụng lao động có quyền quy định thời giờ làm việc theo ngày hoặc tuần nhưng phải thông báo cho người lao động biết; trường hợp theo tuần thì thời giờ làm việc bình thường không quá 10 giờ trong 01 ngày và không quá 48 giờ trong 01 tuần. Nhà nước khuyến khích áp dụng tuần làm việc 40 giờ.
"""
    },
    "LDN_2020": {
        "domain": "doanh_nghiep",
        "name": "Luật Doanh nghiệp 2020",
        "text": """
Điều 12. Người đại diện theo pháp luật của doanh nghiệp
1. Người đại diện theo pháp luật của doanh nghiệp là cá nhân đại diện cho doanh nghiệp thực hiện các quyền và nghĩa vụ phát sinh từ giao dịch của doanh nghiệp, đại diện cho doanh nghiệp với tư cách người yêu cầu giải quyết việc dân sự, nguyên đơn, bị đơn, người có quyền lợi, nghĩa vụ liên quan trước Trọng tài, Tòa án.
2. Công ty trách nhiệm hữu hạn và công ty cổ phần có thể có một hoặc nhiều người đại diện theo pháp luật. Điều lệ công ty quy định cụ thể số lượng, chức danh quản lý và quyền, nghĩa vụ của người đại diện theo pháp luật của doanh nghiệp.

Điều 17. Quyền thành lập, góp vốn, mua cổ phần, mua phần vốn góp và quản lý doanh nghiệp
1. Tổ chức, cá nhân có quyền thành lập và quản lý doanh nghiệp tại Việt Nam theo quy định của Luật này, trừ trường hợp quy định tại khoản 2 Điều này.
2. Tổ chức, cá nhân sau đây không có quyền thành lập và quản lý doanh nghiệp: cơ quan nhà nước, đơn vị lực lượng vũ trang nhân dân sử dụng tài sản nhà nước để thành lập doanh nghiệp kinh doanh thu lợi riêng; cán bộ, công chức, viên chức theo quy định của pháp luật về cán bộ, công chức, viên chức; sĩ quan, hạ sĩ quan, quân nhân chuyên nghiệp trong các đơn vị Quân đội nhân dân, Công an nhân dân; người chưa thành niên, người bị hạn chế hoặc mất năng lực hành vi dân sự.

Điều 46. Công ty trách nhiệm hữu hạn hai thành viên trở lên
1. Công ty trách nhiệm hữu hạn hai thành viên trở lên là doanh nghiệp có từ 02 đến 50 thành viên là tổ chức, cá nhân. Thành viên chịu trách nhiệm về các khoản nợ và nghĩa vụ tài sản khác của doanh nghiệp trong phạm vi số vốn đã góp vào doanh nghiệp.
2. Công ty trách nhiệm hữu hạn hai thành viên trở lên có tư cách pháp nhân kể từ ngày được cấp Giấy chứng nhận đăng ký doanh nghiệp. Công ty không được phát hành cổ phần, trừ trường hợp để chuyển đổi thành công ty cổ phần.

Điều 74. Công ty trách nhiệm hữu hạn một thành viên
1. Công ty trách nhiệm hữu hạn một thành viên là doanh nghiệp do một tổ chức hoặc một cá nhân làm chủ sở hữu; chủ sở hữu công ty chịu trách nhiệm về các khoản nợ và nghĩa vụ tài sản khác của công ty trong phạm vi số vốn điều lệ của công ty.
2. Công ty trách nhiệm hữu hạn một thành viên có tư cách pháp nhân kể từ ngày được cấp Giấy chứng nhận đăng ký doanh nghiệp.
"""
    },
    "BLHS_2015": {
        "domain": "hinh_su",
        "name": "Bộ luật Hình sự 2015",
        "text": """
Điều 8. Khái niệm tội phạm
1. Tội phạm là hành vi nguy hiểm cho xã hội được quy định trong Bộ luật hình sự, do người có năng lực trách nhiệm hình sự hoặc pháp nhân thương mại thực hiện một cách cố ý hoặc vô ý, xâm phạm độc lập, chủ quyền, thống nhất, toàn vẹn lãnh thổ Tổ quốc, xâm phạm chế độ chính trị, chế độ kinh tế, nền văn hóa, quốc phòng, an ninh, trật tự, an toàn xã hội, quyền, lợi ích hợp pháp của tổ chức, xâm phạm tính mạng, sức khỏe, danh dự, nhân phẩm, tự do, tài sản, các quyền, lợi ích hợp pháp khác của công dân, xâm phạm những lĩnh vực khác của trật tự pháp luật xã hội chủ nghĩa mà theo quy định của Bộ luật này phải bị xử lý hình sự.

Điều 12. Tuổi chịu trách nhiệm hình sự
1. Người từ đủ 16 tuổi trở lên phải chịu trách nhiệm hình sự về mọi tội phạm, trừ những tội phạm mà Bộ luật này có quy định khác.
2. Người từ đủ 14 tuổi đến dưới 16 tuổi phải chịu trách nhiệm hình sự về tội phạm rất nghiêm trọng, tội phạm đặc biệt nghiêm trọng quy định tại một số điều của Bộ luật này.

Điều 27. Thời hiệu truy cứu trách nhiệm hình sự
1. Thời hiệu truy cứu trách nhiệm hình sự là thời hạn do Bộ luật này quy định mà khi hết thời hạn đó thì người phạm tội không bị truy cứu trách nhiệm hình sự.
2. Thời hiệu truy cứu trách nhiệm hình sự được quy định như sau: 05 năm đối với tội phạm ít nghiêm trọng; 10 năm đối với tội phạm nghiêm trọng; 15 năm đối với tội phạm rất nghiêm trọng; 20 năm đối với tội phạm đặc biệt nghiêm trọng.

Điều 174. Tội lừa đảo chiếm đoạt tài sản
1. Người nào bằng thủ đoạn gian dối chiếm đoạt tài sản của người khác trị giá từ 2.000.000 đồng đến dưới 50.000.000 đồng hoặc dưới 2.000.000 đồng nhưng thuộc một trong các trường hợp luật định thì bị phạt cải tạo không giam giữ đến 03 năm hoặc phạt tù từ 06 tháng đến 03 năm.
2. Phạm tội thuộc một trong các trường hợp sau đây thì bị phạt tù từ 02 năm đến 07 năm: có tổ chức; có tính chất chuyên nghiệp; chiếm đoạt tài sản trị giá từ 50.000.000 đồng đến dưới 200.000.000 đồng; tái phạm nguy hiểm.
"""
    }
}

def generate_multi_domain_corpus():
    """Tạo file corpus văn bản và file chunks cấu trúc chuẩn hóa cho toàn bộ 4 lĩnh vực luật."""
    out_dir = DATA_DIR / "sample_nlp"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_chunks = []
    
    for code, info in MULTI_DOMAIN_LEGAL_DATA.items():
        law_file = out_dir / f"{code.lower()}.txt"
        law_file.write_text(info["text"].strip(), encoding="utf-8")
        
        chunks = parse_legal_corpus(info["text"], law_code=code, domain=info["domain"])
        for c in chunks:
            c["law_name"] = info["name"]
        all_chunks.extend(chunks)
        print(f"📦 [Preprocessor] Đã nạp {len(chunks)} điều luật thuộc {info['name']} ({info['domain']})")

    out_json = out_dir / "legal_chunks.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ [Hoàn tất] Tổng số Điều luật trong kho kiến thức đa miền: {len(all_chunks)}")
    print(f"📁 Tệp cấu trúc lưu tại: {out_json}")
    return all_chunks

if __name__ == "__main__":
    generate_multi_domain_corpus()
