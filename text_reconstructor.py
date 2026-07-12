"""Spatial line and paragraph reconstruction for EasyOCR regions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Iterable


@dataclass(frozen=True)
class TextRegion:
    text: str
    x: float
    y: float
    width: float
    height: float

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass
class TextLine:
    regions: list[TextRegion] = field(default_factory=list)

    @property
    def x(self) -> float:
        return min(region.x for region in self.regions)

    @property
    def y(self) -> float:
        return min(region.y for region in self.regions)

    @property
    def bottom(self) -> float:
        return max(region.bottom for region in self.regions)

    @property
    def height(self) -> float:
        return self.bottom - self.y

    @property
    def center_y(self) -> float:
        return (self.y + self.bottom) / 2

    @property
    def text(self) -> str:
        return " ".join(region.text.strip() for region in sorted(self.regions, key=lambda item: item.x)
                        if region.text.strip())


class TextReconstructor:
    """Rebuild document flow using region coordinates, without language-model inference."""

    LIST_PATTERN = re.compile(r"^\s*(?:[-*•▪◦‣]|(?:\d+|[A-Za-z가-힣])[.)])(?:\s+|$)")
    SENTENCE_END = re.compile(r"[.!?。！？:：;；]$|(?:다|요|음|함)[.]?$")

    def reconstruct(self, items: Iterable[Any]) -> str:
        regions = [region for item in items if (region := self._to_region(item)) is not None]
        if not regions:
            return ""
        lines = self._group_lines(regions)
        return self._merge_paragraphs(lines)

    def _group_lines(self, regions: list[TextRegion]) -> list[TextLine]:
        typical_height = median(region.height for region in regions)
        lines: list[TextLine] = []
        for region in sorted(regions, key=lambda item: (item.center_y, item.x)):
            candidates = [line for line in lines if self._same_line(line, region, typical_height)]
            if candidates:
                min(candidates, key=lambda line: abs(line.center_y - region.center_y)).regions.append(region)
            else:
                lines.append(TextLine([region]))
        return sorted(lines, key=lambda line: (line.y, line.x))

    @staticmethod
    def _same_line(line: TextLine, region: TextRegion, typical_height: float) -> bool:
        overlap = max(0.0, min(line.bottom, region.bottom) - max(line.y, region.y))
        overlap_ratio = overlap / max(1.0, min(line.height, region.height))
        center_limit = max(typical_height, line.height, region.height) * 0.55
        return overlap_ratio >= 0.35 or abs(line.center_y - region.center_y) <= center_limit

    def _merge_paragraphs(self, lines: list[TextLine]) -> str:
        if not lines:
            return ""
        positive_gaps = [max(0.0, lines[index + 1].y - lines[index].bottom)
                         for index in range(len(lines) - 1)]
        typical_gap = median(positive_gaps) if positive_gaps else 0.0
        typical_height = median(line.height for line in lines)
        paragraphs = [lines[0].text]
        for index, current in enumerate(lines[:-1]):
            following = lines[index + 1]
            gap = max(0.0, following.y - current.bottom)
            if self._paragraph_break(current, following, gap, typical_gap, typical_height, lines, index):
                paragraphs.append("\n" + following.text)
            else:
                paragraphs.append(" " + following.text)
        return re.sub(r"[ \t]+", " ", "".join(paragraphs)).strip()

    def _paragraph_break(self, current: TextLine, following: TextLine, gap: float,
                         typical_gap: float, typical_height: float,
                         lines: list[TextLine], index: int) -> bool:
        if self.LIST_PATTERN.match(current.text) or self.LIST_PATTERN.match(following.text):
            return True
        large_gap = gap > max(typical_height * 1.15, typical_gap * 1.8)
        if large_gap:
            return True
        if self._looks_like_title(current, gap, typical_gap, typical_height):
            return True
        # A pronounced indentation after a completed sentence commonly starts a paragraph.
        left_margin = median(line.x for line in lines)
        indented = following.x - left_margin > typical_height * 1.8
        return indented and bool(self.SENTENCE_END.search(current.text.strip()))

    @staticmethod
    def _looks_like_title(line: TextLine, following_gap: float,
                          typical_gap: float, typical_height: float) -> bool:
        short = len(line.text.replace(" ", "")) <= 24
        prominent = line.height >= typical_height * 1.18
        isolated = following_gap > max(typical_gap * 1.4, typical_height * 0.75)
        return short and (prominent or isolated) and not line.text.endswith((".", "。", ","))

    @staticmethod
    def _to_region(item: Any) -> TextRegion | None:
        text, box = getattr(item, "text", ""), getattr(item, "box", None)
        if not text or not box:
            return None
        try:
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
        except (TypeError, ValueError, IndexError):
            return None
        width, height = max(xs) - min(xs), max(ys) - min(ys)
        if width <= 0 or height <= 0:
            return None
        return TextRegion(str(text), min(xs), min(ys), width, height)
