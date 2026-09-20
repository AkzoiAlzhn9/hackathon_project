# -*- coding: utf-8 -*-
"""Word lists and cheap morphology for Kazakh/Russian spoken-form tokens.

The CRF learns word identities on its own; these lexicons exist so that *rare or
unseen* inflected forms still land on the right side of a decision boundary.
Everything here is lowercase Cyrillic, matching the contest input.
"""

from functools import lru_cache
from typing import FrozenSet, Set


def _s(words: str) -> FrozenSet[str]:
    return frozenset(w for w in words.split() if w)


# --- numerals -------------------------------------------------------------

NUM_RU = _s(
    """
    ноль нуль один одна одно одного одному одним однем два две двух двум двумя
    три трех трём трёх трем тремя четыре четырех четырём четырёх четырем четырьмя
    пять пяти пятью шесть шести шестью семь семи семью восемь восьми восемью восьмью
    девять девяти девятью десять десяти десятью
    одиннадцать одиннадцати двенадцать двенадцати тринадцать тринадцати
    четырнадцать четырнадцати пятнадцать пятнадцати шестнадцать шестнадцати
    семнадцать семнадцати восемнадцать восемнадцати девятнадцать девятнадцати
    двадцать двадцати тридцать тридцати сорок сорока
    пятьдесят пятидесяти шестьдесят шестидесяти семьдесят семидесяти
    восемьдесят восьмидесяти девяносто девяноста
    сто ста сот двести двухсот тремста триста трехсот трёхсот четыреста четырехсот
    пятьсот пятисот шестьсот шестисот семьсот семисот восемьсот восьмисот
    девятьсот девятисот
    тысяча тысячи тысяч тысяче тысячу тысячей тысячами
    миллион миллиона миллионов миллиону миллионе
    миллиард миллиарда миллиардов триллион триллиона триллионов
    полтора полторы полутора пол половина половиной половины
    """
)

NUM_KK = _s(
    """
    нөл ноль бір екі үш төрт бес алты жеті сегіз тоғыз он жиырма отыз қырық елу
    алпыс жетпіс сексен тоқсан жүз мың миллион миллиард триллион жарты жарым
    """
)

NUMBERS: FrozenSet[str] = NUM_RU | NUM_KK

# Multipliers: their presence usually keeps a numeric span going.
MULTIPLIERS = _s(
    """
    тысяча тысячи тысяч тысяче тысячу тысячей миллион миллиона миллионов
    миллиард миллиарда миллиардов триллион триллиона триллионов
    мың миллион миллиард триллион жүз
    """
)

# --- ordinals -------------------------------------------------------------

ORDINALS = _s(
    """
    первый первая первое первого первом первых первым вторый второй вторая второе
    второго втором третий третья третье третьего третьем четвертый четвёртый
    четвертая четвёртая четвертое четвёртое пятый пятая пятое шестой шестая шестое
    седьмой седьмая седьмое восьмой восьмая восьмое девятый девятая девятое
    десятый десятая десятое одиннадцатый двенадцатый тринадцатый четырнадцатый
    пятнадцатый шестнадцатый семнадцатый восемнадцатый девятнадцатый двадцатый
    тридцатый сороковой пятидесятый шестидесятый семидесятый восьмидесятый
    девяностый сотый двухсотый трехсотый тысячный миллионный
    бірінші екінші үшінші төртінші бесінші алтыншы жетінші сегізінші тоғызыншы
    оныншы жиырмасыншы отызыншы қырқыншы елуінші алпысыншы жетпісінші сексенінші
    тоқсаныншы жүзінші мыңыншы соңғы
    """
)

# Russian ordinal stems: cover unseen gender/case endings (пятидесятого, ...).
ORDINAL_STEMS = (
    "первы",
    "перво",
    "перва",
    "вторы",
    "второ",
    "втора",
    "треть",
    "четверт",
    "четвёрт",
    "пят",
    "шест",
    "седьм",
    "восьм",
    "девят",
    "десят",
    "надцат",
    "двадцат",
    "тридцат",
    "сороков",
    "сот",
    "тысячн",
)

# Kazakh ordinal endings (-ыншы/-інші/-ншы/-нші/-ыншi).
ORDINAL_SUFFIX_KK = ("ыншы", "інші", "ншы", "нші", "інщі", "ыншi")

# --- units, currency, percent --------------------------------------------

UNITS = _s(
    """
    метр метра метров метре сантиметр сантиметра сантиметров миллиметр миллиметра
    километр километра километров дециметр гектар гектара гектаров
    грамм грамма граммов килограмм килограмма килограммов тонна тонны тонн
    литр литра литров миллилитр миллилитра кубометр
    ватт ватта ваттов киловатт киловатта вольт вольта ампер ампера
    байт байта байтов килобайт мегабайт гигабайт терабайт мегабит гигабит
    герц гигагерц мегагерц градус градуса градусов
    процент процента процентов проценте промилле пункт пункта пунктов
    метрлік шақырым шақырымға шақырымдық келі келілік келіден
    литрлік граммдық тонналық гектарлық пайыз пайызы пайызға пайыздық
    градустық шаршы текше дана данасы
    """
)

CURRENCY = _s(
    """
    тенге теңге тенгеге доллар доллара долларов долларға евро еуро рубль рубля
    рублей рублях юань юаня юаней фунт фунта фунтов сум сума сом сома гривна
    гривен копейка копеек цент цента центов тиын тиынға манат лира иена вон
    """
)

MEASURE_WORDS: FrozenSet[str] = UNITS | CURRENCY

# --- time / date ----------------------------------------------------------

TIME_WORDS = _s(
    """
    час часа часов часу часам минута минуты минут минуту минутам секунда секунды
    секунд сутки суток неделя недели недель месяц месяца месяцев
    сағат сағатта сағаттан сағатқа минут минутта секунд апта аптада ай айда
    утра вечера дня ночи полдень полночь полдня
    таңғы кешкі түнгі түскі таңертең кешке
    """
)

MONTHS = _s(
    """
    январь января январе февраль февраля феврале март марта марте апрель апреля
    апреле май мая мае июнь июня июне июль июля июле август августа августе
    сентябрь сентября сентябре октябрь октября октябре ноябрь ноября ноябре
    декабрь декабря декабре
    қаңтар қаңтарда ақпан ақпанда наурыз наурызда сәуір сәуірде мамыр мамырда
    маусым маусымда шілде шілдеде тамыз тамызда қыркүйек қыркүйекте қазан қазанда
    қараша қарашада желтоқсан желтоқсанда
    """
)

YEAR_WORDS = _s(
    """
    год года году годом годе лет году годов
    жыл жылы жылдың жылға жылда жылдан жылдары
    """
)

# --- email ----------------------------------------------------------------

EMAIL_MARKERS = _s(
    """
    собачка собака собаки табачка эт этт ат белгісі белгиси дог ит
    нүкте нукте точка тчк точку точкой нүктесі
    подчеркивание подчёркивание нижнее дефис тире сызықша слеш слэш дробь
    астына сызық
    """
)

# Dictated addresses spell the domain out letter by letter ("ка зет" = .kz,
# "эр у" = .ru).  Individually these are ordinary short words, so they are only
# useful together with the markers above -- the CRF weighs that combination.
SPELLED_LETTERS = _s(
    """
    а бэ вэ ге дэ е жэ зэ зет и ка эл эм эн о пэ эр эс тэ у эф ха цэ че ша
    эй эйт би си ди эйч джей кей эль оу пи кью ар ти ви дабл экс уай зед зэт зэд
    """
)

EMAIL_DOMAINS = _s(
    """
    гмайл гмаил джимейл мэйл майл мейл яндекс рамблер аутлук хотмейл инбокс
    бк лист ком кз ру орг нет инфо еду гов юкей мейлру
    """
)

EMAIL_CONTEXT: FrozenSet[str] = EMAIL_MARKERS | EMAIL_DOMAINS


def is_spelled_letter(token: str) -> bool:
    return token in SPELLED_LETTERS

# --- decimals -------------------------------------------------------------

DECIMAL_MARKERS = _s(
    """
    целых целая целое целых десятых десятая сотых сотая тысячных тысячная
    запятая запятой точка бүтін ондық жүздік мыңдық үтір
    """
)

# --- Kazakh morphology ----------------------------------------------------

_KK_SUFFIXES = (
    "дардың", "дердің", "тардың", "тердің", "лардың", "лердің",
    "дарға", "дерге", "тарға", "терге", "ларға", "лерге",
    "дың", "дің", "тың", "тің", "ның", "нің",
    "дан", "ден", "тан", "тен", "нан", "нен",
    "ға", "ге", "қа", "ке", "на", "не",
    "да", "де", "та", "те", "нда", "нде",
    "ды", "ді", "ты", "ті", "ны", "ні",
    "мен", "бен", "пен", "дай", "дей", "тай", "тей",
    "сы", "сі", "ы", "і", "лық", "лік", "дық", "дік", "тық", "тік",
)


@lru_cache(maxsize=1 << 20)
def kk_stem(token: str) -> str:
    """Strip one common Kazakh case/possessive suffix (longest match first)."""
    for suffix in _KK_SUFFIXES:
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _in(token: str, vocabulary: FrozenSet[str]) -> bool:
    return token in vocabulary or kk_stem(token) in vocabulary


def is_number(token: str) -> bool:
    return _in(token, NUMBERS)


def is_multiplier(token: str) -> bool:
    return _in(token, MULTIPLIERS)


def is_ordinal(token: str) -> bool:
    if _in(token, ORDINALS):
        return True
    if any(token.endswith(suffix) for suffix in ORDINAL_SUFFIX_KK) and len(token) > 5:
        return True
    if len(token) > 5 and any(stem in token for stem in ORDINAL_STEMS):
        # Russian ordinals end in adjectival endings; numerals do not.
        return token.endswith(
            ("ый", "ий", "ой", "ая", "яя", "ое", "ее", "ого", "его", "ом", "ем", "ую", "ых", "им", "ым")
        )
    return False


def is_measure_word(token: str) -> bool:
    return _in(token, MEASURE_WORDS)


def is_currency(token: str) -> bool:
    return _in(token, CURRENCY)


def is_time_word(token: str) -> bool:
    return _in(token, TIME_WORDS)


def is_month(token: str) -> bool:
    return _in(token, MONTHS)


def is_year_word(token: str) -> bool:
    return _in(token, YEAR_WORDS)


def is_email_context(token: str) -> bool:
    return token in EMAIL_CONTEXT


def is_decimal_marker(token: str) -> bool:
    return _in(token, DECIMAL_MARKERS)


def is_latin_ish(token: str) -> bool:
    """Cyrillic-only input, so this only flags stray characters."""
    return any("a" <= ch <= "z" for ch in token)


@lru_cache(maxsize=1 << 20)
def tags(token: str) -> FrozenSet[str]:
    """All lexicon tags that fire for a token (used as CRF features).

    Cached: the same token is looked up once per position in the +-2 window, and
    vocabularies are small and highly repetitive.
    """
    out: Set[str] = set()
    if is_number(token):
        out.add("NUM")
    if is_multiplier(token):
        out.add("MULT")
    if is_ordinal(token):
        out.add("ORD")
    if is_measure_word(token):
        out.add("UNIT")
    if is_currency(token):
        out.add("CUR")
    if is_time_word(token):
        out.add("TIMEW")
    if is_month(token):
        out.add("MONTH")
    if is_year_word(token):
        out.add("YEARW")
    if is_email_context(token):
        out.add("MAIL")
    if is_decimal_marker(token):
        out.add("DEC")
    if is_spelled_letter(token):
        out.add("LET")
    return frozenset(out)
