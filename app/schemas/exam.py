from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import AliasChoices, BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from app.schemas.common import Difficulty, QuestionType

# MTF: one block per (type, difficulty) row, holding N pairs under a shared
# `instruction`. Upper bound aligns with QuestionTypeSpec.numberOfQuestions ≤ 50.
MTF_MAX_MATCH_PAIRS = 50


def _int_like(v: Any) -> int:
    """Accept JSON integers or digit strings (e.g. \"1\")."""
    if isinstance(v, bool):
        raise ValueError("boolean is not a valid id")
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            raise ValueError("empty id string")
        try:
            return int(s)
        except ValueError as exc:
            raise ValueError("expected a digit string or integer") from exc
    raise TypeError("expected int or str")


IntLike = Annotated[int, BeforeValidator(_int_like)]


class QuestionTypeSpec(BaseModel):
    numberOfQuestions: int = Field(ge=1, le=50)
    type: QuestionType
    difficultyLevel: Difficulty


class ExamPayload(BaseModel):
    """Exam scope: class, subject, publication (or legacy `book_type`), and one or more `chapters`."""

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _chapters_from_legacy_chapter(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        has_chapters = d.get("chapters") not in (None, [], ())
        if not has_chapters and d.get("chapter") is not None:
            ch = d["chapter"]
            d["chapters"] = ch if isinstance(ch, list) else [ch]
        return d

    class_id: IntLike = Field(alias="class", description="Class id; integer or digit string; must match upload `class`.")
    subject: IntLike = Field(description="Subject id; integer or digit string; must match upload `subject`.")
    publication: IntLike = Field(
        validation_alias=AliasChoices("publication", "book_type"),
        description="Publication id; integer or digit string; same as upload `publication` (legacy JSON key `book_type` accepted).",
    )
    chapters: list[IntLike] = Field(
        min_length=1,
        max_length=40,
        description="One or more chapter numbers to pull context from (metadata `chapter`). Use a single element for one chapter.",
    )
    questionTypes: list[QuestionTypeSpec] = Field(
        min_length=1,
        max_length=32,
        description="At least one (type, difficulty, count) block; drives how many questions the model should output.",
    )
    description: str = Field(
        default="",
        description="Teacher notes/instructions to customize the exam style, focus areas, tone, or constraints.",
    )

    @field_validator("chapters", mode="after")
    @classmethod
    def _unique_sorted_chapters(cls, v: list[int]) -> list[int]:
        return sorted(set(v))


class SourceCitation(BaseModel):
    """RAG source shape (not yet part of ``ExamResponse`` questions; kept for future APIs / tooling)."""

    book_id: str
    chapter: int
    pages: list[int] = Field(min_length=1)


class QuestionOption(BaseModel):
    optionLabel: str
    text: str
    displayOrder: int


class MatchPair(BaseModel):
    pairKey: str = Field(
        default="",
        description="Short label for the pair (A, B, C, ...); assigned by the server per displayOrder.",
    )
    leftText: str = Field(description="Left-column item (Column A).")
    rightText: str = Field(default="", description="Right-column item (Column B) that matches `leftText`.")
    displayOrder: int


class MTFInnerQuestion(BaseModel):
    questionCode: str
    displayOrder: int
    matchPairs: list[MatchPair] = Field(min_length=1, max_length=MTF_MAX_MATCH_PAIRS)


class MCQQuestion(BaseModel):
    type: Literal["MCQ"] = "MCQ"
    questionCode: str
    difficulty: Difficulty
    text: str
    displayOrder: int
    options: list[QuestionOption] = Field(min_length=1)
    correctOption: str = Field(
        default="",
        description="The `optionLabel` of the single correct choice (e.g. A).",
    )


class TOFQuestion(BaseModel):
    type: Literal["TOF"] = "TOF"
    questionCode: str
    difficulty: Difficulty
    text: str
    displayOrder: int
    answer: bool = Field(
        default=False,
        description="True if the statement in `text` is correct; false if it is incorrect.",
    )


class FIBQuestion(BaseModel):
    type: Literal["FIB"] = "FIB"
    questionCode: str
    difficulty: Difficulty
    text: str
    displayOrder: int
    answers: list[str] = Field(
        default_factory=list,
        description="Correct fill for each blank in order (aligned with _____ placeholders in `text`).",
    )


class MTFQuestion(BaseModel):
    """Match-the-following block.

    One MTF block per (type, difficulty) row in the payload. The block carries a
    shared `instruction` heading and one inner question holding all `matchPairs`
    for that row. `numberOfQuestions` from the payload row equals
    `questions[0].matchPairs` length.
    """

    type: Literal["MTF"] = "MTF"
    difficulty: Difficulty
    instruction: str = Field(
        default="Match the items in Column A with Column B.",
        description="Shared heading shown above the table of pairs.",
    )
    questions: list[MTFInnerQuestion] = Field(
        min_length=1,
        max_length=1,
        description="Exactly one inner question holding the N matchPairs for this MTF row.",
    )


class DESQuestion(BaseModel):
    type: Literal["DES"] = "DES"
    questionCode: str
    difficulty: Difficulty
    text: str
    displayOrder: int
    modelAnswer: str = Field(
        default="",
        description="Marking reference / model answer (may include rubric bullets merged from the model).",
    )

    @model_validator(mode="before")
    @classmethod
    def _des_text_from_answer(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if str(d.get("text") or "").strip():
            return d
        ma = d.get("modelAnswer")
        if ma is not None and str(ma).strip():
            d["text"] = str(ma).strip()
            return d
        return d


Question = Annotated[
    Union[MCQQuestion, TOFQuestion, FIBQuestion, MTFQuestion, DESQuestion],
    Field(discriminator="type"),
]


class ExamResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    generated_at: datetime
    class_: str = Field(
        validation_alias=AliasChoices("class", "grade"),
        serialization_alias="class",
    )
    subject: str
    publication: str = Field(
        default="",
        description="Publication id as digit string (same scope as the generate request).",
    )
    chapters: list[int] = Field(
        default_factory=list,
        description="Chapter numbers from the generate request (metadata `chapter` scope).",
    )
    description: str = Field(
        default="",
        description="Merged prose: exam summary (model), exam analytics (model), then teacher instructions from the request (last).",
    )
    totalQuestions: int
    questions: list[Question]
