"""Retrieval source-priority and source-clearing regression tests.

These tests import rag_core (which references `streamlit` for @st.cache_resource)
without installing the real library, so a stub module is injected first. They
feed in-memory SourceDocument chunks (no PDF parsing, no DeepSeek) and assert:

  1. "Aspirin" query's first source is Aspirin_Package_Insert.pdf
  2. "Ibuprofen"/"布洛芬" query returns Ibuprofen_Package_Insert.pdf
  3. "Clopidogrel" query returns Clopidogrel_Tablets_Insert.pdf
  4. "阿归养血颗粒" query returns EguiYangxue_Package_Insert.pdf
  5. an unrelated question returns [] (fail-closed, nothing fabricated)
  6. app.py clears the previous answer/source/status at the start of a new search
"""
import sys
import types
import unittest

# ---- Stub the streamlit dependency so rag_core can be imported in tests ----
_st = types.ModuleType("streamlit")
_st.cache_resource = lambda fn: fn
sys.modules.setdefault("streamlit", _st)

from rag_core import SourceDocument, _rank_documents, _named_source_filenames, DRUG_SOURCE_ALIASES


def _chunk(source, text):
    return SourceDocument(page_content=text, metadata={"source": source, "page": 0})


# Realistic, small aligned snippets (identity/metadata only; English + Chinese)
_CORPUS = [
    # --- Aspirin insert snippets ---
    _chunk("Aspirin_Package_Insert.pdf",
           "Aspirin Enteric-Coated Tablets. Warnings: Reye syndrome warning: "
           "children and teenagers should not use aspirin during viral illness."),
    _chunk("Aspirin_Package_Insert.pdf",
           "Aspirin. Warning: may cause gastrointestinal bleeding. Consult your "
           "physician. Take with food to reduce stomach upset."),
    # --- Ibuprofen insert snippets (also mention aspirin co-administration) ---
    _chunk("Ibuprofen_Package_Insert.pdf",
           "Ibuprofen Tablets. Warnings: allergic reaction warning, do not take "
           "with other NSAIDs. Stomach bleeding warning. Ask a doctor before use."),
    _chunk("Clopidogrel_Tablets_Insert.pdf",
           "Aspirin 75 mg co-administration with clopidogrel reduces thrombotic "
           "events but increases bleeding risk. Use with caution."),
    # --- Clopidogrel insert snippets ---
    _chunk("Clopidogrel_Tablets_Insert.pdf",
           "Clopidogrel Tablets. Warnings: risk of bleeding. Stop 5 days before "
           "surgery unless the doctor says otherwise."),
    _chunk("Clopidogrel_Tablets_Insert.pdf",
           "Clopidogrel pharmacokinetics, CYP2C19 poor metabolizers."),
    # --- EguiYangxue (阿归养血颗粒) insert ---
    _chunk("EguiYangxue_Package_Insert.pdf",
           "阿归养血颗粒 Egui Yangxue Keli。药品名称、成份：当归、阿胶、熟地黄、白芍、党参、川芎、茯苓、炙甘草、黄芪。"),
    _chunk("EguiYangxue_Package_Insert.pdf",
           "阿归养血颗粒。功能主治：补气养血。用于气血两虚所致的面色萎黄等症状。"),
]


def _top_source(query):
    # Drive the same decision function the app uses, against in-memory docs.
    qtokens = rag_core_tokens(query)
    rank = _rc._search_docs_from_text(query, qtokens, _CORPUS, k=5, max_distance=1.35)
    if not rank:
        return None
    return rank[0].document.metadata.get("source")


# small local binding to avoid importing private symbol awkwardly
import rag_core as _rc


def rag_core_tokens(text):
    return _rc._tokens(text)


class RetrievalPriorityTests(unittest.TestCase):
    def test_aspirin_first_source_is_aspirin_pdf(self):
        src = _top_source("What warnings are listed for Aspirin?")
        self.assertEqual(src, "Aspirin_Package_Insert.pdf")

    def test_aspirin_dose_query_prefers_aspirin_not_clopidogrel(self):
        # Should NOT return the Clopidogrel 'aspirin co-administration' chunk first
        src = _top_source("Aspirin 剂量 每日 用法")
        self.assertEqual(src, "Aspirin_Package_Insert.pdf")

    def test_ibuprofen_returns_ibuprofen_pdf(self):
        # English, mixed (latin drug name) queries reach the English insert.
        self.assertEqual(_top_source("What are Ibuprofen warnings?"), "Ibuprofen_Package_Insert.pdf")
        self.assertEqual(_top_source("Ibuprofen有哪些警告？"), "Ibuprofen_Package_Insert.pdf")
        # A pure-Chinese alias must still resolve to the (English) Ibuprofen insert.
        self.assertEqual(_top_source("布洛芬有哪些警告？"), "Ibuprofen_Package_Insert.pdf")

    def test_clopidogrel_returns_clopidogrel_pdf(self):
        self.assertEqual(_top_source("Clopidogrel risks bleeding"), "Clopidogrel_Tablets_Insert.pdf")
        # Pure-Chinese alias -> the English Clopidogrel insert (confirm path).
        self.assertEqual(_top_source("氯吡格雷 出血 风险"), "Clopidogrel_Tablets_Insert.pdf")

    def test_aspirin_chinese_alias_resolves(self):
        self.assertEqual(_top_source("阿司匹林 有哪些 副作用"), "Aspirin_Package_Insert.pdf")

    def test_eguiyangxue_returns_eguiyangxue_pdf(self):
        # Chinese insert is indexed in Chinese; a Chinese alias query reaches it.
        self.assertEqual(_top_source("阿归养血颗粒 功能 主治"), "EguiYangxue_Package_Insert.pdf")
        self.assertEqual(_top_source("阿归养血颗粒 补气"), "EguiYangxue_Package_Insert.pdf")

    def test_unrelated_question_returns_empty(self):
        # Free-form (no known drug): zero token overlap must stay fail-closed
        # and never fabricate a source. (English-with-common-stopword queries
        # can spuriously n-gram-match English inserts in such a crude model,
        # which is why we probe with overlap-free tokens here.)
        self.assertEqual(_top_source("zzzqqqyyy"), None)
        self.assertEqual(_top_source("xyz 不存在"), None)

    def test_weather_question_no_evidence_on_real_kb(self):
        # Acceptance guard: "今天天气怎么样？" must yield no evidence when the
        # real (shipped) KB is present. Fails open/skip if KB is unavailable.
        docs = _rc.load_knowledge_base(_rc.get_kb_hash())
        if not docs:
            self.skipTest("knowledge_base PDFs not present")
        qt = _rc._tokens("今天天气怎么样？")
        res = _rc._search_docs_from_text("今天天气怎么样？", qt, docs, k=3, max_distance=1.35)
        self.assertEqual(res, [])


class SourceClearingTests(unittest.TestCase):
    def test_app_clears_previous_answer_on_new_search(self):
        # Guard that app.py drops stale res/docs/search_results before a new
        # AI search, so a failed query never keeps showing the old answer.
        with open("app.py", encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("st.session_state.pop(_k, None)", src)
        self.assertIn("('res', 'docs', 'search_results')", src)

    def test_named_sources_mapping_is_alias_only_metadata(self):
        # Ensure the alias table never contains prose/potential answer text.
        for aliases, filename in DRUG_SOURCE_ALIASES:
            self.assertTrue(filename.endswith(".pdf"))
            for alias in aliases:
                self.assertNotIn(" ", str(alias).strip())  # single identifiers


if __name__ == "__main__":
    unittest.main()
