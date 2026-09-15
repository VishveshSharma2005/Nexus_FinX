"""Generate synthetic demo documents into corpus/samples/.

Everything produced here is invented. No real borrower, lender, or agreement is
involved, and every page carries a visible synthetic marker so a generated file
can never be mistaken for a customer document.

The content is deliberately seeded with the problems FinX exists to catch, so
that later phases have something real to find:

* penal interest added to the rate of interest, which the penal-charges
  circular prohibits;
* a foreclosure charge and a lock-in on a floating-rate personal loan;
* a unilateral tenor-extension clause on rate reset;
* charges in the agreement that the Key Facts Statement does not disclose,
  and an interest rate that disagrees with it.

Two agreement versions are written so that content-hash versioning has a real
before/after to diff.

Usage:  python scripts/make_sample_documents.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "corpus" / "samples"

SYNTHETIC_BANNER = "SYNTHETIC DEMO DOCUMENT - NOT A REAL LOAN AGREEMENT"

PAGE_WIDTH, PAGE_HEIGHT = fitz.paper_size("a4")
MARGIN = 56
BODY_WIDTH = PAGE_WIDTH - 2 * MARGIN

STYLES = {
    "title": {"size": 15, "font": "hebo", "space_after": 16},
    "heading": {"size": 10.5, "font": "hebo", "space_after": 6},
    "body": {"size": 9.5, "font": "helv", "space_after": 10},
    "small": {"size": 8, "font": "helv", "space_after": 8},
}


class Writer:
    """Minimal top-down text flow with page breaks."""

    def __init__(self, document: fitz.Document) -> None:
        self.document = document
        self.page: fitz.Page | None = None
        self.y = 0.0
        self._new_page()

    def _new_page(self) -> None:
        self.page = self.document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        self.page.insert_text(
            (MARGIN, MARGIN - 22),
            SYNTHETIC_BANNER,
            fontsize=7,
            fontname="hebo",
            color=(0.65, 0.16, 0.16),
        )
        self.y = MARGIN

    def write(self, text: str, style: str = "body") -> None:
        spec = STYLES[style]
        size = spec["size"]
        # Estimate height, then grow the box until the text fits.
        height = 0.0
        for _ in range(3):
            height = self._estimate_height(text, size)
            if self.y + height > PAGE_HEIGHT - MARGIN:
                self._new_page()
            rect = fitz.Rect(MARGIN, self.y, MARGIN + BODY_WIDTH, self.y + height + size)
            overflow = self.page.insert_textbox(
                rect, text, fontsize=size, fontname=spec["font"], align=0
            )
            if overflow >= 0:
                break
            size -= 0.0  # keep size; loop grows the estimate instead
            text_lines = self._estimate_height(text, size)
            height = text_lines * 1.5
        self.y += height + spec["space_after"]

    @staticmethod
    def _estimate_height(text: str, size: float) -> float:
        chars_per_line = max(20, int(BODY_WIDTH / (size * 0.5)))
        lines = 0
        for paragraph in text.split("\n"):
            lines += max(1, -(-len(paragraph) // chars_per_line))
        return lines * (size * 1.45) + size


def build_pdf(path: Path, blocks: list[tuple[str, str]]) -> None:
    document = fitz.open()
    writer = Writer(document)
    for style, text in blocks:
        writer.write(text, style)
    document.save(path)
    document.close()


LENDER = "Meridian Finserv Private Limited"
LENDER_LINE = (
    f"{LENDER}, a Non-Banking Financial Company registered with the Reserve Bank of India, "
    "having its registered office at 4th Floor, Anand Business Park, Ahmedabad 380015 "
    '(hereinafter the "Lender")'
)
BORROWER_LINE = (
    "Mr. Rakesh Bhavsar, an individual residing at 22 Silver Oak Residency, Vadodara 390007, "
    'holding Aadhaar ending 4417 (hereinafter the "Borrower")'
)


def agreement_blocks(*, version: int, agreement_date: str) -> list[tuple[str, str]]:
    """Clause text for the personal loan agreement.

    Version 2 changes exactly three clauses, so the version diff has a known
    expected answer: the foreclosure charge, the penal charge, and the reset
    clause. Everything else must hash identically and be reused, not
    re-embedded.
    """
    is_v2 = version == 2

    prepayment = (
        "6.1 The Borrower may prepay the Loan in part or in full at any time after the "
        "expiry of a lock-in period of twelve (12) months from the date of first "
        "disbursement. Any prepayment made shall attract a foreclosure charge of 4% "
        "(four per cent) of the principal outstanding, plus applicable taxes. No "
        "prepayment shall be permitted out of funds borrowed from another lender."
        if not is_v2
        else "6.1 The Borrower may prepay the Loan in part or in full at any time, "
        "without any lock-in period. No foreclosure or prepayment charge shall be "
        "levied on this Loan. Prepayment may be made irrespective of the source of "
        "funds used by the Borrower."
    )

    penal = (
        "7.1 In the event of delay in payment of any instalment, penal interest at the "
        "rate of 24% (twenty four per cent) per annum shall be added to the rate of "
        "interest applicable to the Loan and shall be compounded monthly until the "
        "default is cured."
        if not is_v2
        else "7.1 In the event of delay in payment of any instalment, a penal charge of "
        "2% (two per cent) per month on the overdue instalment amount shall be levied "
        "as a fixed charge. Such penal charge shall not be added to the rate of "
        "interest and shall not be capitalised."
    )

    reset = (
        "8.2 Upon any reset of the benchmark rate, the Lender shall be entitled at its "
        "sole discretion to extend the tenure of the Loan or to increase the amount of "
        "the Equated Monthly Instalment, or both, without prior intimation to or "
        "consent of the Borrower."
        if not is_v2
        else "8.2 Upon any reset of the benchmark rate, the Lender shall communicate to "
        "the Borrower the revised Equated Monthly Instalment and tenure. The Borrower "
        "shall have the option to switch to a fixed rate of interest, to enhance the "
        "Equated Monthly Instalment, to elongate the tenure, or any combination "
        "thereof, and shall be informed of the charges applicable to such switch."
    )

    return [
        ("title", "PERSONAL LOAN AGREEMENT"),
        (
            "small",
            f"Agreement reference MFS/PL/2026/{'0' if not is_v2 else '1'}88214    "
            f"Version {version}    Executed on {agreement_date}",
        ),
        (
            "body",
            f"THIS PERSONAL LOAN AGREEMENT is made on {agreement_date} at Ahmedabad, "
            f"BETWEEN {LENDER_LINE} AND {BORROWER_LINE}.",
        ),
        ("heading", "1. DEFINITIONS AND INTERPRETATION"),
        (
            "body",
            '1.1 "Loan" means the personal loan facility of Rs. 5,00,000 (Rupees Five '
            "Lakh only) sanctioned by the Lender to the Borrower on the terms set out "
            "in this Agreement and in the Schedule annexed hereto.",
        ),
        (
            "body",
            '1.2 "EMI" means the Equated Monthly Instalment payable by the Borrower '
            "comprising principal and interest, as specified in the Schedule.",
        ),
        (
            "body",
            '1.3 "Benchmark Rate" means the external benchmark to which the rate of '
            "interest on the Loan is linked, as notified by the Lender from time to time.",
        ),
        ("heading", "2. AMOUNT AND DISBURSEMENT"),
        (
            "body",
            "2.1 The Lender agrees to lend and the Borrower agrees to borrow the sum of "
            "Rs. 5,00,000 (Rupees Five Lakh only), repayable over a tenure of 48 "
            "(forty eight) months.",
        ),
        (
            "body",
            "2.2 The Loan shall be disbursed to the Borrower's designated bank account "
            "after deduction of the charges set out in Clause 4 and the insurance "
            "premium set out in Clause 5.",
        ),
        ("heading", "3. INTEREST"),
        (
            "body",
            "3.1 The Loan shall carry interest at a floating rate of 14.50% (fourteen "
            "point five zero per cent) per annum, linked to the Benchmark Rate, "
            "calculated on a monthly reducing balance basis.",
        ),
        (
            "body",
            "3.2 The Lender may revise the spread over the Benchmark Rate upon annual "
            "review of the Borrower's credit profile.",
        ),
        ("heading", "4. FEES AND CHARGES"),
        (
            "body",
            "4.1 A processing fee of 2% (two per cent) of the sanctioned amount, plus "
            "applicable Goods and Services Tax, shall be deducted from the disbursement.",
        ),
        (
            "body",
            "4.2 Documentation and stamping charges of Rs. 2,500 shall be payable by the Borrower.",
        ),
        (
            "body",
            "4.3 A loan administration charge of Rs. 500 per annum shall be recovered "
            "annually on the anniversary of the disbursement date.",
        ),
        (
            "body",
            "4.4 Cheque or mandate dishonour charges of Rs. 750 per instance shall be "
            "payable, in addition to any charges levied by the Borrower's bank.",
        ),
        ("heading", "5. INSURANCE"),
        (
            "body",
            "5.1 The Borrower shall obtain a credit life insurance policy covering the "
            "outstanding amount of the Loan. The premium of Rs. 12,400 shall be "
            "deducted from the disbursement and financed as part of the Loan.",
        ),
        ("heading", "6. PREPAYMENT AND FORECLOSURE"),
        ("body", prepayment),
        (
            "body",
            "6.2 Part prepayment shall be permitted only in multiples of the EMI amount "
            "and not more than twice in any financial year.",
        ),
        ("heading", "7. DEFAULT AND PENAL CHARGES"),
        ("body", penal),
        (
            "body",
            "7.2 Upon the occurrence of an event of default, the Lender may recall the "
            "entire outstanding amount of the Loan and enforce any security furnished "
            "by the Borrower.",
        ),
        ("heading", "8. RESET OF FLOATING RATE"),
        (
            "body",
            "8.1 The rate of interest shall be reset at intervals of three months from "
            "the date of first disbursement, in accordance with movements in the "
            "Benchmark Rate.",
        ),
        ("body", reset),
        ("heading", "9. RECOVERY"),
        (
            "body",
            "9.1 The Lender may engage third party recovery agents for the purpose of "
            "recovering amounts due under this Agreement. The Borrower shall bear the "
            "costs of such recovery, subject to a maximum of Rs. 5,000 per instance.",
        ),
        ("heading", "10. GOVERNING LAW AND DISPUTE RESOLUTION"),
        (
            "body",
            "10.1 This Agreement shall be governed by the laws of India, and the courts "
            "at Ahmedabad shall have exclusive jurisdiction. Any dispute shall be "
            "referred to arbitration by a sole arbitrator appointed by the Lender.",
        ),
        ("heading", "SCHEDULE - PARTICULARS OF THE LOAN"),
        (
            "body",
            "Sanctioned amount: Rs. 5,00,000\n"
            "Tenure: 48 months\n"
            "Rate of interest: 14.50% per annum, floating\n"
            "Equated Monthly Instalment: Rs. 13,780\n"
            "Processing fee: 2% plus applicable taxes\n"
            "Insurance premium financed: Rs. 12,400\n"
            f"Foreclosure charge: {'Nil' if is_v2 else '4% of principal outstanding'}\n"
            f"Lock-in period: {'Nil' if is_v2 else '12 months from first disbursement'}",
        ),
    ]


def kfs_blocks() -> list[tuple[str, str]]:
    """Key Facts Statement that deliberately disagrees with the agreement.

    The rate, the insurance premium and the foreclosure charge differ from the
    v1 agreement, which is what the Phase 7 audit must surface.
    """
    return [
        ("title", "KEY FACTS STATEMENT"),
        (
            "small",
            "Part 1 - Interest rate and fees/charges    "
            "Applicant: Mr. Rakesh Bhavsar    Date: 8 February 2026",
        ),
        ("heading", "1. LOAN PROPOSAL DETAILS"),
        (
            "body",
            f"1.1 Name of the Regulated Entity: {LENDER}\n"
            "1.2 Type of loan: Personal loan\n"
            "1.3 Sanctioned loan amount (in Rupees): 5,00,000\n"
            "1.4 Disbursal schedule: 100% upfront\n"
            "1.5 Loan term: 48 months",
        ),
        ("heading", "2. INSTALMENT DETAILS"),
        (
            "body",
            "2.1 Type of instalment: Equated Monthly Instalment\n"
            "2.2 Number of instalments: 48\n"
            "2.3 Amount of each instalment (in Rupees): 13,588",
        ),
        ("heading", "3. INTEREST RATE AND TYPE"),
        (
            "body",
            "3.1 Annual rate of interest: 13.90% per annum\n"
            "3.2 Type of interest: Floating\n"
            "3.3 Reference benchmark and spread: External benchmark plus 5.15%",
        ),
        ("heading", "4. FEES AND CHARGES"),
        (
            "body",
            "4.1 Processing fee: Rs. 10,000 (2% of sanctioned amount), plus applicable taxes\n"
            "4.2 Insurance charges: Rs. 8,900\n"
            "4.3 Documentation charges: Rs. 2,500\n"
            "4.4 Foreclosure charges: Nil\n"
            "4.5 Charges for switching from floating to fixed rate: Rs. 2,000",
        ),
        ("heading", "5. ANNUAL PERCENTAGE RATE"),
        (
            "body",
            "5.1 Annual Percentage Rate (APR): 15.20% per annum\n"
            "5.2 The APR includes the rate of interest and all fees and charges "
            "recovered by the Regulated Entity, including charges recovered on behalf "
            "of third parties.",
        ),
        ("heading", "6. CONTINGENT CHARGES"),
        (
            "body",
            "6.1 Penal charges on delayed payment: 2% per month on the overdue instalment\n"
            "6.2 Cheque or mandate dishonour charges: Rs. 750 per instance\n"
            "6.3 Other charges: Nil",
        ),
        ("heading", "7. GRIEVANCE REDRESS"),
        (
            "body",
            "7.1 Nodal grievance redressal officer: Ms. Priya Nair, "
            "grievance@meridianfinserv.example, 1800-000-0000\n"
            "7.2 If the complaint is not resolved within 30 days, the complainant may "
            "approach the RBI Complaint Management System portal.",
        ),
        ("heading", "8. VALIDITY"),
        ("body", "8.1 This Key Facts Statement is valid until 15 February 2026."),
    ]


SAMPLES = {
    "personal_loan_agreement_v1.pdf": lambda: agreement_blocks(
        version=1, agreement_date="10 February 2026"
    ),
    "personal_loan_agreement_v2.pdf": lambda: agreement_blocks(
        version=2, agreement_date="4 May 2026"
    ),
    "personal_loan_kfs.pdf": kfs_blocks,
    # Sanctioned before 1 January 2026, so the 2025 pre-payment Directions do
    # not govern it, and the earlier circulars that did are not in the corpus.
    # Exists to exercise that coverage gap in the browser.
    "personal_loan_agreement_2025.pdf": lambda: agreement_blocks(
        version=1, agreement_date="18 November 2025"
    ),
}


def main(argv: list[str] | None = None) -> int:
    """Write every sample, or only the ones named on the command line.

    Rewriting a file that already exists changes its bytes -- the PDF carries a
    creation timestamp -- and so its content-derived document id. Name only the
    files you mean to regenerate.
    """
    wanted = set(argv or [])
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for filename, builder in SAMPLES.items():
        if wanted and filename not in wanted:
            continue
        path = SAMPLES_DIR / filename
        build_pdf(path, builder())
        print(f"wrote {path.relative_to(REPO_ROOT)}  ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
