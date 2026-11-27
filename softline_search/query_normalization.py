import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from .catalog import Offer


_TOKEN = re.compile(r"[^\W_]+(?:[-._/][^\W_]+)*\+?", re.UNICODE)
_LAYOUT_WORD = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\[\];',.`]*")
_ENGLISH = "`qwertyuiop[]asdfghjkl;'zxcvbnm,."
_RUSSIAN = "ёйцукенгшщзхъфывапролджэячсмитьбю"
_TO_RUSSIAN = str.maketrans({
    **dict(zip(_ENGLISH, _RUSSIAN)),
    **{key.upper(): value.upper() for key, value in zip(_ENGLISH, _RUSSIAN) if key.isalpha()},
})
_TO_ENGLISH = str.maketrans(_RUSSIAN + _RUSSIAN.upper(), _ENGLISH + _ENGLISH.upper())
_FUNCTION_WORDS = set("и в на для с по под до от за из без не но или к у а о об при как где кто это чтоб чтобы".split())
_FUNCTION_WORDS.update("a an the and or for of at with without to on in pro standard premium home".split())
_FUNCTION_WORDS.update("полтинника восьмерку терабайт желательно предпочтительно дороже дешевле минимум максимум".split())

_ALIASES = {
    "антивирь": "антивирус", "антивир": "антивирус", "фаервол": "межсетевой экран",
    "файрвол": "межсетевой экран", "бэкап": "резервное копирование",
    "бэкапов": "резервного копирования", "прога": "программа", "софт": "программы",
    "почтовик": "почтовый клиент", "файломенеджер": "файловый менеджер",
    "графредактор": "графический редактор", "снесённые": "удалённые", "видосов": "видео",
    "вайтборд": "интерактивная доска", "рендера": "визуализации", "запила": "создания",
    "рэковый": "стоечный", "сервак": "сервер", "сервака": "сервера",
    "ноут": "ноутбук", "ноута": "ноутбука", "ноуты": "ноутбуки",
    "моник": "монитор", "видюха": "видеокарта", "видюху": "видеокарту",
    "клава": "клавиатура", "мышка": "мышь", "вебка": "веб-камера",
    "оперативка": "оперативная память", "оперативку": "оперативная память",
    "проц": "процессор", "бп": "блок питания", "бесперебойник": "источник бесперебойного питания",
    "бесперебойника": "источника бесперебойного питания", "аккум": "аккумулятор",
    "свитч": "коммутатор", "локалки": "локальной сети", "сетевуха": "сетевая карта",
    "вайфай": "Wi-Fi", "вай фай": "Wi-Fi", "вайфайный свисток": "адаптер Wi-Fi",
    "хард": "жесткий диск", "ссдшка": "SSD", "ссд": "SSD", "м2": "M.2",
    "жоский": "жесткий", "провод": "кабель", "недарагая": "недорогая",
    "ддр 4": "DDR4", "ддр4": "DDR4", "ддр 5": "DDR5", "ддр5": "DDR5",
    "системник": "системный блок", "системника": "системного блока",
    "комп": "компьютер", "компа": "компьютера", "компу": "компьютеру",
    "телик": "телевизор", "4джи": "4G LTE", "хдмай": "HDMI",
    "тсд": "терминал сбора данных", "лицуха": "лицензия",
    "бессрочка": "бессрочная лицензия", "винда": "Windows", "винрар": "WinRAR",
    "астра линукс": "Astra Linux", "тимвивер": "TeamViewer", "энидеск": "AnyDesk",
    "майкрософт": "Microsoft", "майкрасофт": "Microsoft", "асус": "ASUS",
    "касперский": "Kaspersky", "каспера": "Kaspersky", "самсунг": "Samsung",
    "сяоми": "Xiaomi", "доктор веб": "Dr.Web", "финридер": "FineReader",
    "радмин": "Radmin", "корел дро": "CorelDRAW", "ворд": "Word", "ексель": "Excel",
    "эксель": "Excel", "крипто про": "КриптоПро", "летучая мышь": "The Bat!",
}


def query_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold().replace("ё", "е")


_ALIASES = {query_key(key): value for key, value in _ALIASES.items()}
_ALIAS_PATTERN = re.compile(
    r"(?<![\w./-])(?:" + "|".join(re.escape(key).replace("е", "[её]") for key in sorted(_ALIASES, key=len, reverse=True))
    + r")(?![\w./-])", re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class QueryChange:
    kind: str
    before: str
    after: str


class QueryNormalizer:
    def __init__(self, offers: tuple[Offer, ...], category_names: list[str]) -> None:
        self.words: Counter[str] = Counter()
        self.codes: dict[str, str] = {}
        for offer in offers:
            self.words.update(query_key(word) for word in _TOKEN.findall(offer.name))
            self.words.update(query_key(word) for word in _TOKEN.findall(offer.vendor))
            if offer.vendor_code:
                self.codes[query_key(offer.vendor_code)] = offer.vendor_code
        for category in category_names:
            self.words.update(query_key(word) for word in _TOKEN.findall(category))
        self.words.update({word: 10 for word in _FUNCTION_WORDS})
        for canonical in _ALIASES.values():
            self.words.update(query_key(word) for word in _TOKEN.findall(canonical))
        self.words.update({key: 10 for key in _ALIASES if " " not in key})

    def suggestion_prefixes(self, query: str) -> tuple[str, ...]:
        normalized, _ = self.normalize(query)
        alphabet = _TO_ENGLISH if any("а" <= char <= "я" for char in query_key(query)) else _TO_RUSSIAN
        return tuple(dict.fromkeys(query_key(text) for text in (normalized, query.translate(alphabet))))

    def normalize(self, query: str) -> tuple[str, tuple[QueryChange, ...]]:
        text = " ".join(unicodedata.normalize("NFC", query).split())
        changes: list[QueryChange] = []

        def apply(before: str, after: str, kind: str) -> str:
            if before != after:
                changes.append(QueryChange(kind, before, after))
            return after

        protected = {
            match.span() for match in _TOKEN.finditer(text)
            if query_key(match.group()) in self.codes or any(char.isdecimal() for char in match.group())
        }
        layout = []
        for match in _LAYOUT_WORD.finditer(text):
            before = match.group()
            key = query_key(before)
            if any(start <= match.start() < end for start, end in protected) or key in self.words:
                continue
            translation = _TO_ENGLISH if any("а" <= char <= "я" for char in key) else _TO_RUSSIAN
            after = before.translate(translation)
            target = query_key(after)
            if after != before and (target in self.words or target in self.codes) and len(key) >= 4:
                layout.append((match.start(), match.end(), before, after))
        if layout:
            left, right = layout[0][0], layout[-1][1]
            for match in re.finditer(r"(?<!\w)b(?!\w)", text):
                if left < match.start() < right:
                    layout.append((*match.span(), match.group(), "и"))
            parts = []
            start = 0
            for left, right, before, after in sorted(layout):
                parts.extend((text[start:left], apply(before, after, "раскладка")))
                start = right
            text = "".join((*parts, text[start:]))

        def aliases(value: str) -> str:
            return _ALIAS_PATTERN.sub(
                lambda match: apply(match.group(), _ALIASES[query_key(match.group())], "словарь"), value
            )

        text = aliases(text)

        def fix(match: re.Match[str]) -> str:
            before = match.group()
            key = query_key(before)
            if key in self.codes or key in self.words or len(key) < 4 or len(key) > 30:
                return before
            candidate = self._best_spelling(key)
            return apply(before, candidate or before, "опечатка")

        return aliases(_TOKEN.sub(fix, text)), tuple(changes)

    def _best_spelling(self, word: str) -> str | None:
        # Pure numbers, model capacities and version numbers have independent meaning.
        digits = any(char.isdecimal() for char in word)
        if digits and (not any(char.isalpha() for char in word) or len(word) < 6):
            return None
        alphabet = _RUSSIAN if any("а" <= char <= "я" for char in word) else "abcdefghijklmnopqrstuvwxyz"
        if digits:
            alphabet += "0123456789"
        candidates = set()
        for i, char in enumerate(word):
            left, right = word[:i], word[i + 1:]
            candidates.add(left + right)
            candidates.update(left + letter + right for letter in alphabet)
            if right:
                candidates.add(left + right[0] + char + right[1:])
        for i in range(len(word) + 1):
            candidates.update(word[:i] + letter + word[i:] for letter in alphabet)
        candidates.discard(word)
        if digits:
            matches = [candidate for candidate in candidates if candidate in self.codes]
            if matches:
                best = min(matches, key=lambda candidate: (-self.words[candidate], candidate))
                return self.codes[best]
            return None
        hits = [candidate for candidate in candidates if self.words[candidate] >= 2]
        return min(hits, key=lambda candidate: (-self.words[candidate], candidate)) if hits else None
