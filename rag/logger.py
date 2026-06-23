import os
from langfuse import Langfuse


def get_langfuse():
    try:
        lf = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://jp.cloud.langfuse.com")
        )
        return lf
    except Exception as e:
        print(f"Langfuse init error: {e}")
        return None


def trace_query(user, question, answer, scores, agentic_mode):
    lf = get_langfuse()
    if not lf:
        return
    try:
        trace_id = lf.create_trace_id()

        with lf.start_observation(
            name="research_query",
            trace_id=trace_id,
            user_id=user,
            input={"question": question},
            output={"answer": answer},
            metadata={"agentic_mode": agentic_mode}
        ):
            lf.score_current_trace(
                name="relevance",
                value=scores["relevance"] / 100
            )
            lf.score_current_trace(
                name="rouge1",
                value=scores["rouge1"] / 100
            )
            lf.score_current_trace(
                name="context_coverage",
                value=scores["context_coverage"] / 100
            )

        lf.flush()
    except Exception as e:
        print(f"Langfuse trace error: {e}")


def trace_document_upload(user, filename, chunks):
    lf = get_langfuse()
    if not lf:
        return
    try:
        trace_id = lf.create_trace_id()

        with lf.start_observation(
            name="document_upload",
            trace_id=trace_id,
            user_id=user,
            input={"filename": filename},
            output={"chunks": chunks}
        ):
            pass

        lf.flush()
    except Exception as e:
        print(f"Langfuse upload error: {e}")