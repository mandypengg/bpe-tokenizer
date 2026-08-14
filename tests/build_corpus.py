"""
Generate tests/corpus.json: 5,000 diverse strings for exercising the tokenizers.

Run it directly to regenerate:

    python tests/build_corpus.py

The corpus is deterministic — same SEED, same 5,000 strings, byte for byte — so
it can be committed and a test failure on case #3172 means the same thing today
as it will next month. Each case carries the category it came from, so a failure
reports "cjk #12" rather than an unlabelled blob of text.

What's in it: English prose, Python/JavaScript/C source, markdown, CJK, Arabic,
Cyrillic, emoji (including ZWJ families, skin-tone modifiers, flags, keycaps and
tag sequences), unusual whitespace, special-token lookalikes, mixed-script text,
and the degenerate cases — the empty string and several hundred single
characters, including every ASCII codepoint from NUL to DEL.

Two things are deliberately excluded:

  * Lone surrogates. They survive json.dumps but not .encode("utf-8"), so every
    tokenizer in this package would fail on them for reasons that have nothing
    to do with BPE.
  * Nothing else — control characters, unpaired ZWJ, unpaired skin-tone
    modifiers and truncated grapheme clusters are all in, on purpose. They are
    valid UTF-8 and a tokenizer has no business caring.

The strings containing "<|endoftext|>" live in their own `special_tokens`
category so that a blanket `decode(encode(t)) == t` test can filter them out:
RegexTokenizer.encode defaults to allowed_special="none_raise" and will raise on
them, which is the correct behaviour, not a bug to work around.
"""

from __future__ import annotations

import json
import random
import textwrap
import unicodedata
from pathlib import Path

SEED = 20260814
TOTAL = 5000
OUT_PATH = Path(__file__).resolve().parent / "corpus.json"

# Build order matters: the corpus is globally deduplicated, and categories
# early in this list get first claim on a string. Degenerate cases come first
# because a single "a" is hard to reach any other way, while prose has an
# effectively unbounded supply of alternatives.
TARGETS = [
    ("degenerate", 350),
    ("whitespace", 400),
    ("special_tokens", 100),
    ("emoji", 600),
    ("cjk", 500),
    ("arabic", 300),
    ("cyrillic", 300),
    ("code_python", 400),
    ("code_javascript", 350),
    ("code_c", 350),
    ("markdown", 400),
    ("english_prose", 600),
    ("mixed_script", 350),
]


# -- source material ----------------------------------------------------------

PROSE = [
    "The quick brown fox jumps over the lazy dog.",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.",
    "She didn't think he'd have come this far, but here we are — 3,000 miles and eleven days later.",
    "\"Elementary,\" said he, \"you know my methods.\"",
    "I can't, won't, shan't and don't — that's four contractions in a row, and they'll all tokenize differently.",
    "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse...",
    "The company's Q3 revenue rose 12.4% to $1.7B, beating analysts' estimates of $1.63B.",
    "Dr. Watson, M.D., arrived at 221B Baker St. at 4:15 p.m. on Tuesday.",
    "THIS SENTENCE IS ENTIRELY IN CAPITAL LETTERS AND SHOULD TOKENIZE POORLY.",
    "this sentence has no capitals at all and no punctuation either",
    "Visit https://example.com/path?query=1&other=2#fragment for more information.",
    "Email me at first.last+tag@sub.domain.co.uk if you have questions.",
    "A well-thought-out, hyphen-heavy, multi-part compound-word example.",
    "In 1969, 3 astronauts travelled 384,400 km in 76 hours.",
    "Rain, rain, go away; come again another day.",
    "The rain in Spain stays mainly in the plain.",
    "He said, 'She said, \"They said it wouldn\\'t work.\"'",
    "Once upon a time, in a land far, far away, there lived a tokenizer who could not count.",
    "Supercalifragilisticexpialidocious antidisestablishmentarianism pneumonoultramicroscopicsilicovolcanoconiosis",
    "To be, or not to be, that is the question:\nWhether 'tis nobler in the mind to suffer\nThe slings and arrows of outrageous fortune.",
    "The meeting is scheduled for 2026-08-14T09:30:00Z, which is 05:30 EDT.",
    "Item 1. Item 2. Item 3. Item 4. Item 5. Item 6. Item 7. Item 8.",
    "naïve café résumé façade jalapeño Zoë piñata Møller Ærø",
    "Der Fluß fließt durch die Straße — großartig, größer, am größten.",
    "L'année dernière, j'ai vu qu'il n'y avait qu'une seule possibilité.",
    "¿Dónde está la biblioteca? ¡Está allí, a la vuelta de la esquina!",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore.",
    "a a a a a a a a a a a a a a a a a a a a a a a a",
    "The     spacing     here     is     deliberately     wide.",
    "Numbers: 0 1 12 123 1234 12345 123456 1234567 12345678 999999999",
]

PY = [
    textwrap.dedent("""\
        def fib(n: int) -> int:
            if n < 2:
                return n
            return fib(n - 1) + fib(n - 2)
        """),
    textwrap.dedent("""\
        class Tokenizer:
            def __init__(self, pattern: str | None = None):
                self.pattern = pattern or DEFAULT
                self.merges: dict[tuple[int, int], int] = {}

            @property
            def vocab_size(self) -> int:
                return len(self.vocab)
        """),
    textwrap.dedent("""\
        @functools.lru_cache(maxsize=None)
        def bytes_to_unicode():
            bs = list(range(ord("!"), ord("~") + 1))
            cs = bs[:]
            n = 0
            for b in range(2**8):
                if b not in bs:
                    bs.append(b)
                    cs.append(2**8 + n)
                    n += 1
            return dict(zip(bs, [chr(c) for c in cs]))
        """),
    'print(f"{name!r} scored {score:.2f}% on {total:,} items")',
    "ids = [list(chunk.encode('utf-8')) for chunk in re.findall(pat, text)]",
    "stats = {p: stats.get(p, 0) + 1 for p in zip(ids, ids[1:])}",
    textwrap.dedent("""\
        try:
            value = int(raw)
        except (ValueError, TypeError) as e:
            logger.warning("bad value %r: %s", raw, e)
            value = None
        finally:
            handle.close()
        """),
    textwrap.dedent("""\
        async def main():
            async with aiohttp.ClientSession() as session:
                results = await asyncio.gather(*(fetch(session, u) for u in urls))
            return [r for r in results if r is not None]
        """),
    "x = {**defaults, **overrides, 'key': [1, 2, 3], ('a', 'b'): {4, 5}}",
    "assert all(idx in tok.vocab for idx in ids), f'bad ids: {ids}'",
    "# TODO(mandy): this is O(n^2) and shows up in the profile\n",
    "if __name__ == '__main__':\n    main()\n",
    "from bpe.base import Tokenizer, get_stats, merge  # noqa: F401",
    "GPT2_SPLIT_PATTERN = r\"\"\"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+\"\"\"",
    "lambda p: ranks.get(p, float('inf'))",
    "matrix = [[0] * cols for _ in range(rows)]",
    "self._cache: dict[str, list[int]] = collections.defaultdict(list)",
    textwrap.dedent("""\
        with open(path, "rb") as f:
            header = f.read(4)
            if header != b"\\x89PNG":
                raise ValueError("not a png")
        """),
    "yield from (x**2 for x in range(10) if x % 3 != 0)",
    "print('tab\\there', 'newline\\nhere', sep='|', end='\\n\\n')",
]

JS = [
    "const sum = (a, b) => a + b;",
    "export default function App({ children, className = '' }) { return <div className={className}>{children}</div>; }",
    textwrap.dedent("""\
        async function fetchAll(urls) {
          const responses = await Promise.all(urls.map((u) => fetch(u)));
          return Promise.all(responses.map((r) => r.json()));
        }
        """),
    "const { a, b: renamed, ...rest } = obj;",
    "const [first, ...others] = arr.filter(Boolean).sort((x, y) => y - x);",
    "console.log(`Hello, ${name}! You have ${count} new ${count === 1 ? 'message' : 'messages'}.`);",
    "if (x ?? y) { obj?.deeply?.nested?.value ??= fallback; }",
    "const re = /^[\\w.+-]+@[\\w-]+\\.[\\w.]{2,}$/giu;",
    textwrap.dedent("""\
        class EventEmitter {
          #listeners = new Map();
          on(event, fn) {
            (this.#listeners.get(event) ?? this.#listeners.set(event, []).get(event)).push(fn);
            return this;
          }
        }
        """),
    "document.querySelectorAll('.item').forEach((el) => el.classList.toggle('active'));",
    "module.exports = { parse, stringify, VERSION: '1.0.0' };",
    "import React, { useState, useEffect, useCallback } from 'react';",
    "setTimeout(() => { throw new Error('boom'); }, 0);",
    "const json = JSON.stringify({ nested: { deep: [1, 2, { x: null }] } }, null, 2);",
    "for (const [key, value] of Object.entries(map)) console.log(key, '=>', value);",
    "let s = 'it\\'s a \"quoted\" string with \\\\ backslashes';",
    "// eslint-disable-next-line no-unused-vars\nconst _unused = 42;",
    "type Result<T, E = Error> = { ok: true; value: T } | { ok: false; error: E };",
    "export const useDebounce = (value, ms = 300) => { const [v, setV] = useState(value); return v; };",
    "window.addEventListener('resize', debounce(handleResize, 100), { passive: true });",
]

C = [
    "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n",
    textwrap.dedent("""\
        int main(int argc, char **argv) {
            if (argc < 2) {
                fprintf(stderr, "usage: %s <file>\\n", argv[0]);
                return EXIT_FAILURE;
            }
            return 0;
        }
        """),
    textwrap.dedent("""\
        typedef struct node {
            int value;
            struct node *next;
        } node_t;
        """),
    "#define MAX(a, b) ((a) > (b) ? (a) : (b))",
    "#define ARRAY_LEN(x) (sizeof(x) / sizeof((x)[0]))",
    "static inline uint32_t rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }",
    textwrap.dedent("""\
        for (size_t i = 0; i < n; i++) {
            buf[i] = (unsigned char)(src[i] ^ key[i % keylen]);
        }
        """),
    "char *p = malloc(len + 1);\nif (!p) { perror(\"malloc\"); abort(); }\nmemcpy(p, src, len);\np[len] = '\\0';",
    "printf(\"%-10s %5d %8.3f %#x %p\\n\", name, count, ratio, flags, (void *)ptr);",
    textwrap.dedent("""\
        switch (state) {
            case STATE_INIT:  /* fall through */
            case STATE_IDLE:
                state = STATE_RUN;
                break;
            default:
                assert(0 && "unreachable");
        }
        """),
    "while ((c = getc(fp)) != EOF) { if (c == '\\n') lines++; }",
    "const char *msgs[] = { \"ok\", \"warn\", \"error\", NULL };",
    "unsigned long long total = 0ULL;\ndouble ratio = (double)bytes / (double)tokens;",
    "/* multi-line comment\n * spanning several lines\n */",
    "#ifdef __cplusplus\nextern \"C\" {\n#endif",
    "struct { int x, y; } point = { .x = 1, .y = 2 };",
    "int (*compare)(const void *, const void *) = &cmp_int;\nqsort(a, n, sizeof(int), compare);",
    "if (fd < 0 && errno == EINTR) continue;",
    "#pragma once\n#include \"internal.h\"\n",
    "uint8_t mask = 0b1010'1010;  // C++14 separators, invalid in plain C",
]

MD = [
    "# Heading 1\n\nSome introductory paragraph text.\n",
    "## Installation\n\n```bash\npip install -r requirements.txt\n```\n",
    "- item one\n- item two\n  - nested item\n  - another nested\n- item three\n",
    "1. First\n2. Second\n3. Third\n   1. Third-a\n   2. Third-b\n",
    "| Column A | Column B | Column C |\n|---------:|:--------:|----------|\n| 1        | two      | three    |\n| 4        | five     | six      |\n",
    "> A blockquote.\n>\n> > Nested deeper.\n",
    "Use `inline_code()` and ``code with ` backtick`` inline.",
    "**bold**, *italic*, ***both***, ~~strikethrough~~, `code`, and _underscores_.",
    "[link text](https://example.com \"title\") and ![image](./img/logo.png)",
    "---\n\n***\n\n___\n",
    "- [x] done task\n- [ ] pending task\n",
    "Footnote reference[^1].\n\n[^1]: The footnote body.\n",
    "```python\ndef f(x):\n    return x * 2\n```\n",
    "```\nplain fenced block, no language\n```\n",
    "<div align=\"center\">\n  <img src=\"logo.svg\" width=\"120\" />\n</div>\n",
    "Term\n: Definition of the term\n",
    "Line ending in two spaces  \nforces a hard break.",
    "---\ntitle: Front matter\ntags: [a, b, c]\n---\n\nBody starts here.\n",
    "See [the spec][spec] for details.\n\n[spec]: https://spec.example.com\n",
    "Escaped characters: \\*not italic\\*, \\_not emphasis\\_, \\# not a heading.",
]

CJK = [
    "你好，世界！",
    "今天天气很好，我们去公园散步吧。",
    "中文分词是自然语言处理中的一个基础任务。",
    "繁體中文與簡體中文在字形上有明顯差異。",
    "他說：「這本書我讀過了。」",
    "北京、上海、广州、深圳是中国的一线城市。",
    "价格是￥1,234.56元，折合约$170。",
    "こんにちは世界",
    "私は日本語を勉強しています。",
    "東京都渋谷区でランチを食べました。",
    "コンピューターとインターネットとプログラミング",
    "ﾊﾝｶｸｶﾀｶﾅ ﾃｽﾄ ﾃﾞｰﾀ",
    "「引用符」と『二重引用符』、それに読点、句点。",
    "ひらがな・カタカナ・漢字・ローマ字 abc の混在",
    "안녕하세요 세계",
    "한국어 토크나이저 테스트입니다.",
    "서울특별시 강남구 테헤란로 123",
    "한글 (첫 글자는 분해된 자모)",
    "ＡＢＣＤＥ１２３４５　全角文字",
    "𠮷野家で𩸽を食べた",  # astral-plane CJK extension B
    "汉字一二三四五六七八九十百千万亿",
    "日本語のテキストは分かち書きをしません",
    "中日韓統一表意文字 / 中日韩统一表意文字",
    "번역기는 문맥을 이해하지 못한다.",
    "この文には、句読点が。たくさん、あります。",
    "龘齉靐龗鱻爨癵驫麤",  # dense, rare, high-stroke characters
    "㊙️㊗️🈚🈯🈲🈳🈵🈴",
    "水曜日、午前10時30分に会議があります。",
]

ARABIC = [
    "مرحبا بالعالم",
    "السلام عليكم ورحمة الله وبركاته",
    "اللغة العربية من أجمل اللغات في العالم.",
    "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
    "الْكِتَابُ عَلَى الطَّاوِلَةِ",
    "١٢٣٤٥٦٧٨٩٠ هي الأرقام العربية الهندية",
    "كــــتــــاب مــــمــــدود بالتطويل",
    "ما هو اسمك؟ اسمي محمد.",
    "نسبة النمو ٪٥ خلال العام الماضي، ثم انخفضت.",
    "البريد الإلكتروني user@example.com والموقع https://example.com",
    "سلام دنیا — این متن فارسی است.",
    "خوش آمدید به صفحه اصلی",
    "اردو ایک خوبصورت زبان ہے۔",
    "الأرقام 2024 مختلطة مع النص العربي هنا.",
    "ذهب الطالب إلى المدرسة، ثم عاد إلى المنزل.",
    "لا إله إلا الله",
    "قال تعالى: ﴿وَقُل رَّبِّ زِدْنِي عِلْمًا﴾",
    "الحروف: أ ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي",
    "همزة القطع أ إ ؤ ئ ء وهمزة الوصل ا",
    "تاء مربوطة ة وتاء مفتوحة ت وألف مقصورة ى",
]

CYRILLIC = [
    "Привет, мир!",
    "Съешь же ещё этих мягких французских булок, да выпей чаю.",
    "В чащах юга жил бы цитрус? Да, но фальшивый экземпляр!",
    "Москва — столица России, население более 12 миллионов человек.",
    "ПРИВЕТ, ЭТО ЗАГЛАВНЫЕ БУКВЫ.",
    "Привіт, світе! Ґедзь їхав на ґанок.",
    "Українська мова має літери ї, є, і та ґ.",
    "Здравей, свят! Това е български текст.",
    "Ђорђе је купио њиве у Шапцу.",
    "Љубав, њежност и ћирилица.",
    "Съёмка ёлки в Ёбурге — ё встречается редко.",
    "Цена: 1 234,56 ₽ (с НДС 20%)",
    "Дата: 14 августа 2026 г., время 09:30",
    "Токенизатор разбивает текст на подслова.",
    "Смешанный Mixed текст Text здесь Here.",
    "Алфавит: абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    "АЛФАВИТ: АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
    "«Ёлки-палки», — сказал он и ушёл.",
    "Не будите спящую собаку — она может укусить.",
    "Электронная почта: пользователь@пример.рф",
]

SPECIAL = [
    "<|endoftext|>",
    "hello<|endoftext|>world",
    "<|endoftext|><|endoftext|>",
    "text before <|endoftext|> text after",
    "<|endoftext|>\n\nnew document starts here",
    "trailing text then the token<|endoftext|>",
    "<|",
    "|>",
    "<|endoftext|",
    "|endoftext|>",
    "< |endoftext| >",
    "<|end|>",
    "<|unknown_special|>",
    "<|fim_prefix|>def f(<|fim_suffix|>):<|fim_middle|>",
    "<|im_start|>user\nhi<|im_end|>",
    "<<|endoftext|>>",
    "\\<|endoftext|\\>",
    "<|endoftext|>[MASK][CLS][SEP][PAD][UNK]",
    "The literal string \"<|endoftext|>\" appears inside quotes.",
    "<|endoftext|> ",
]


# -- generated pools ----------------------------------------------------------

ZWJ = "‍"
VS16 = "️"
KEYCAP = "⃣"
TONES = [chr(cp) for cp in range(0x1F3FB, 0x1F400)]  # 5 skin-tone modifiers

_EMOJI_SINGLE = list(
    "😀😃😄😁😆😅🤣😂🙂🙃😉😊😇🥰😍🤩😘😗😚😙🥲😋😛😜🤪😝🤑🤗🤭🤫🤔"
    "🐶🐱🐭🐹🐰🦊🐻🐼🐨🐯🦁🐮🐷🐸🐵🦄🐝🐛🦋🐌🐞🐜🕷🦂🐢🐍🦎🦖🦕🐙🦑"
    "🍏🍎🍐🍊🍋🍌🍉🍇🍓🫐🍈🍒🍑🥭🍍🥥🥝🍅🍆🥑🥦🥬🥒🌶🫑🌽🥕🫒🧄🧅"
    "⚽🏀🏈⚾🥎🎾🏐🏉🥏🎱🪀🏓🏸🏒🏑🥍🏏🪃🥅⛳🪁🏹🎣🤿🥊🥋🎽🛹🛼🛷⛸"
    "🚗🚕🚙🚌🚎🏎🚓🚑🚒🚐🛻🚚🚛🚜🦯🦽🦼🛴🚲🛵🏍🛺🚨🚔🚍🚘🚖🚡🚠🚟"
)
_SKINNABLE = list("👋🤚🖐✋🖖👌🤌🤏✌🤞🫰🤟🤘🤙👈👉👆🖕👇☝🫵👍👎✊👊🤛🤜👏🙌🫶👐🤲🙏")
_PEOPLE = ["👩", "👨", "🧑"]
_ROLES = ["🚀", "💻", "🔬", "🍳", "🌾", "🏫", "⚕" + VS16, "⚖" + VS16, "🎨", "🔧", "✈" + VS16, "🚒"]
_FLAG_CODES = [
    "US", "GB", "JP", "DE", "FR", "BR", "IN", "CN", "KR", "MX", "ZA", "NG", "EG", "CA",
    "AU", "RU", "IT", "ES", "SE", "NO", "FI", "PL", "TR", "AR", "CL", "PE", "KE", "GH",
    "VN", "TH", "PH", "ID", "MY", "SG", "NZ", "IE", "PT", "GR", "IL", "AE",
]


def _emoji_pool() -> list[str]:
    """Emoji, weighted towards the multi-codepoint sequences that break things."""
    out: list[str] = []
    out += _EMOJI_SINGLE
    out += _SKINNABLE

    # base + skin tone modifier
    for base in _SKINNABLE:
        out += [base + tone for tone in TONES]

    # person + ZWJ + role, with and without a tone on the person
    for person in _PEOPLE:
        for role in _ROLES:
            out.append(person + ZWJ + role)
            out += [person + tone + ZWJ + role for tone in TONES]

    # families and couples: the long ZWJ chains
    out += [
        "👨" + ZWJ + "👩" + ZWJ + "👧" + ZWJ + "👦",
        "👩" + ZWJ + "👩" + ZWJ + "👦",
        "👨" + ZWJ + "👨" + ZWJ + "👧" + ZWJ + "👧",
        "👩" + ZWJ + "👦",
        "👨" + ZWJ + "👧",
        "🧑" + ZWJ + "🧑" + ZWJ + "🧒",
        "👩" + ZWJ + "❤" + VS16 + ZWJ + "👨",
        "👨" + ZWJ + "❤" + VS16 + ZWJ + "💋" + ZWJ + "👨",
        "👩" + ZWJ + "❤" + VS16 + ZWJ + "💋" + ZWJ + "👩",
        "🧑" + TONES[2] + ZWJ + "🤝" + ZWJ + "🧑" + TONES[4],
        "👩" + TONES[0] + ZWJ + "🦰",
        "👨" + TONES[3] + ZWJ + "🦳",
        "🧑" + ZWJ + "🦼",
        "🏳" + VS16 + ZWJ + "🌈",
        "🏳" + VS16 + ZWJ + "⚧" + VS16,
        "🐻" + ZWJ + "❄" + VS16,
        "👁" + VS16 + ZWJ + "🗨" + VS16,
        "🏴" + ZWJ + "☠" + VS16,
    ]

    # regional-indicator flags
    out += ["".join(chr(0x1F1E6 + ord(c) - 65) for c in code) for code in _FLAG_CODES]

    # tag-sequence subdivision flags
    for sub in ("gbeng", "gbsct", "gbwls"):
        out.append("🏴" + "".join(chr(0xE0000 + ord(c)) for c in sub) + "\U000e007f")

    # keycaps
    out += [c + VS16 + KEYCAP for c in "0123456789#*"]

    # text vs emoji presentation of the same base codepoint
    for base in "☀☁☂☃❄✂✈✉✏":
        out += [base, base + VS16, base + "︎"]

    # emoji embedded in text, where the regex split pattern has to deal with them
    out += [
        "I 💙 tokenizers",
        "great work 👍🏽 thanks!",
        "😀😀😀😀😀😀😀😀",
        "one 🍎 two 🍎🍎 three 🍎🍎🍎",
        "👨‍👩‍👧‍👦 is one grapheme and seven codepoints",
        "trailing emoji 🎉",
        "🎉 leading emoji",
        "emoji🎉without🎉spaces",
    ]
    return out


_WS_CHARS = [
    "\t", "\n", "\r", "\r\n", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x1f", "\x85",
    " ", " ", " ", " ", " ", " ", " ", " ",
    " ", " ", " ", " ", " ", " ", " ", " ",
    " ", " ", "　",
    "​",  # zero-width space: not whitespace to Unicode, looks like it to humans
    "﻿",  # BOM / zero-width no-break space
    "᠎",  # Mongolian vowel separator, whitespace only in older Unicode
]

# The GPT-2 split pattern has a `\s+(?!\S)` branch and a ` ?\p{L}+` branch, so
# where whitespace sits relative to a word changes how the whole run chunks.
_WS_TEMPLATES = [
    "a{ws}b",
    "{ws}word",
    "word{ws}",
    "{ws}",
    "line one{ws}line two",
    "def f():{ws}pass",
    "{ws}mid{ws}dle{ws}",
]


def _whitespace_pool() -> list[str]:
    out: list[str] = []
    for ws in _WS_CHARS:
        for count in (1, 2, 3):
            run = ws * count
            for template in _WS_TEMPLATES:
                out.append(template.format(ws=run))
    out += [
        "        eight leading spaces",
        "\t\t\ttab indented",
        "mixed \t     spacing",
        "trailing spaces then newline   \n",
        "\n\n\n\n",
        "\r\n\r\n",
        "line\rline\nline\r\nline",
        "﻿BOM at the very start",
        "word​with​zero​width​spaces",
        " " * 64,
        "\t" * 16,
        " \n \n \n ",
    ]
    return out


_DEGENERATE_UNICODE = list(
    "áéíóúàèìòùâêîôûäëïöüãõñçßøåæœþðıİĳŀŉſ"
    "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω"
    "אבגדהוזחטיכלמנסעפצקרשת"
    "अआइईउऊएऐओऔकखगघङचछजझ"
    "กขคงจฉชซฌญ"
    "ကခဂဃငစဆဇ"
    "ႠႡႢႣႤႥ"
    "ԱԲԳԴԵԶ"
    "←↑→↓↔↕⇐⇒⇔∀∂∃∅∇∈∉∋∏∑√∝∞∠∧∨∩∪∫≈≠≡≤≥⊂⊃⊆⊇⊕⊗⊥⋅"
    "€£¥¢₹₽₩₪₫₴₦₡₱฿"
    "①②③④⑤ⅠⅡⅢⅣⅤ½⅓¼¾⅞"
    "©®™§¶†‡•‰′″‹›«»„“”‘’—–…"
)

_DEGENERATE_CLUSTERS = [
    "👨" + ZWJ + "👩" + ZWJ + "👧" + ZWJ + "👦",
    "👍" + TONES[3],
    "🇺🇸",
    "1" + VS16 + KEYCAP,
    "é",           # combining acute
    "à́̂",  # stacked combining marks
    "กำ",      # Thai with vowel sign
    "क्ष",  # Devanagari conjunct क्ष
    "한",  # decomposed Hangul 한
    "q̣̇",     # combining marks that reorder under NFC
    "ᄒ",
    "Å",      # A + ring above == Å
    "🏳" + VS16 + ZWJ + "🌈",
    "سَّ",     # Arabic letter + shadda + fatha
    "אָ",      # Hebrew alef + qamats
    "𝔸", "𝕬", "𝓐", "𝐀", "𝟘", "𝜋",
    "🀄", "🂡", "🃏",
    "\U0001d11e",        # G clef
    "\U0002000b",        # astral CJK
    "\U000e0041",        # tag latin capital A, on its own
    ZWJ, VS16, TONES[0], "͏",  # unpaired joiners and modifiers, on purpose
]


def _degenerate_pool(rng: random.Random, want: int) -> list[str]:
    pool = [""]
    pool += [chr(i) for i in range(0x00, 0x80)]  # NUL through DEL, controls included
    pool += _DEGENERATE_UNICODE
    pool += _DEGENERATE_CLUSTERS
    # pad with assigned codepoints swept from across the BMP and beyond
    while len(pool) < want + 100:
        cp = rng.randrange(0x20, 0x30000)
        if 0xD800 <= cp <= 0xDFFF:  # surrogates cannot be encoded as UTF-8
            continue
        ch = chr(cp)
        if unicodedata.category(ch) == "Cn":  # unassigned
            continue
        pool.append(ch)
    return pool


# -- mutators -----------------------------------------------------------------
# Each takes (rng, pool) and returns a new string. They exist so that a few dozen
# handwritten fragments per category can fan out to hundreds of distinct cases
# without the result degenerating into noise.


def _mut_slice(rng: random.Random, pool: list[str]) -> str:
    """A slice cut at arbitrary offsets — through words, quotes, ZWJ chains."""
    s = rng.choice(pool)
    if len(s) < 2:
        return s
    i = rng.randrange(len(s))
    j = rng.randint(i + 1, len(s))
    return s[i:j]


def _mut_join(rng: random.Random, pool: list[str]) -> str:
    joiner = rng.choice(["", " ", "\n", "\n\n", "\t", ", ", " — ", " | "])
    return joiner.join(rng.choice(pool) for _ in range(rng.randint(2, 3)))


def _mut_pad(rng: random.Random, pool: list[str]) -> str:
    lead = rng.choice(["", " ", "  ", "\t", "\n", "   ", " \t ", " "])
    trail = rng.choice(["", " ", "\n", "  ", "\t\n", " "])
    return lead + rng.choice(pool) + trail


def _mut_repeat(rng: random.Random, pool: list[str]) -> str:
    return rng.choice(pool) * rng.randint(2, 4)


def _mut_words(rng: random.Random, pool: list[str]) -> str:
    """A run of consecutive words, so prose fragments start on word boundaries."""
    words = rng.choice(pool).split(" ")
    if len(words) < 2:
        return rng.choice(pool)
    i = rng.randrange(len(words))
    return " ".join(words[i : i + rng.randint(1, 8)])


DEFAULT_MUTATORS = [_mut_slice, _mut_join, _mut_pad, _mut_repeat]
TEXT_MUTATORS = DEFAULT_MUTATORS + [_mut_words]


# -- assembly -----------------------------------------------------------------


def _grow(
    name: str,
    pool: list[str],
    n: int,
    seen: set[str],
    mutators: list | None = None,
) -> list[str]:
    """Take n strings not already in `seen`, from the pool then from mutations."""
    rng = random.Random(f"{SEED}:{name}")
    out: list[str] = []

    def take(s: str) -> None:
        if s not in seen:
            seen.add(s)
            out.append(s)

    for s in pool:
        if len(out) >= n:
            break
        take(s)

    if len(out) < n and mutators:
        limit = max(50_000, n * 500)
        for _ in range(limit):
            if len(out) >= n:
                break
            take(rng.choice(mutators)(rng, pool))

    if len(out) != n:
        raise RuntimeError(
            f"category {name!r}: only produced {len(out)} of {n} unique strings; "
            "widen the pool or add mutators"
        )
    return out


def build(seen: set[str], name: str, n: int) -> list[str]:
    if name == "degenerate":
        pool = _degenerate_pool(random.Random(f"{SEED}:degenerate:pool"), n)
        return _grow(name, pool, n, seen)
    if name == "whitespace":
        return _grow(name, _whitespace_pool(), n, seen, DEFAULT_MUTATORS)
    if name == "emoji":
        return _grow(name, _emoji_pool(), n, seen, DEFAULT_MUTATORS)
    if name == "mixed_script":
        # deliberately cross script boundaries inside a single chunk
        pool = PROSE + CJK + ARABIC + CYRILLIC + _EMOJI_SINGLE + MD[:5] + PY[:5]
        return _grow(name, pool, n, seen, [_mut_join, _mut_join, _mut_pad, _mut_slice])

    pool = {
        "special_tokens": SPECIAL,
        "cjk": CJK,
        "arabic": ARABIC,
        "cyrillic": CYRILLIC,
        "code_python": PY,
        "code_javascript": JS,
        "code_c": C,
        "markdown": MD,
        "english_prose": PROSE,
    }[name]
    return _grow(name, pool, n, seen, TEXT_MUTATORS)


def build_corpus() -> dict:
    seen: set[str] = set()
    cases = []
    for name, n in TARGETS:
        for text in build(seen, name, n):
            cases.append({"category": name, "text": text})

    assert len(cases) == TOTAL, f"expected {TOTAL} cases, got {len(cases)}"
    return {
        "version": 1,
        "seed": SEED,
        "count": len(cases),
        "categories": dict(TARGETS),
        "cases": cases,
    }


def validate(data: dict) -> None:
    """Every case must be UTF-8 encodable and survive a JSON round trip."""
    for i, case in enumerate(data["cases"]):
        text = case["text"]
        assert isinstance(text, str), f"case {i} is not a string"
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as e:  # a lone surrogate slipped in
            raise AssertionError(f"case {i} ({case['category']}) is not UTF-8: {e}")

    texts = [c["text"] for c in data["cases"]]
    assert len(set(texts)) == len(texts), "duplicate strings in corpus"

    reloaded = json.loads(json.dumps(data, ensure_ascii=False))
    assert reloaded == data, "corpus does not survive a JSON round trip"


def load_corpus(path: Path = OUT_PATH) -> list[dict]:
    """Read the generated corpus; each case is {"category": str, "text": str}."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)["cases"]


def main() -> None:
    data = build_corpus()
    validate(data)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.write("\n")

    lengths = [len(c["text"]) for c in data["cases"]]
    nbytes = sum(len(c["text"].encode("utf-8")) for c in data["cases"])
    print(f"wrote {OUT_PATH} — {data['count']} cases, "
          f"{nbytes:,} UTF-8 bytes, {OUT_PATH.stat().st_size:,} bytes on disk")
    print(f"lengths: min {min(lengths)}, max {max(lengths)}, "
          f"mean {sum(lengths) / len(lengths):.1f} codepoints\n")
    for name, n in TARGETS:
        sample = next(c["text"] for c in data["cases"] if c["category"] == name)
        print(f"  {name:<16} {n:>5}   e.g. {sample[:56]!r}")


if __name__ == "__main__":
    main()
