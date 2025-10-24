from langchain_community.document_loaders import JSONLoader
import os
import json
from pathlib import Path

def load_json(file_path: str):
    """
    Loads a JSON file and returns its content as a list of documents.
    """
    # If your JSON fields are not plain text (for example lists or objects),
    # set text_content=False so the loader will keep the raw value instead of
    # forcing it into the `page_content` string field and raising the error.
    # Use `to_entries[]` so we keep the original object key (e.g. the course
    # title) together with its value. Each entry returned will be an object
    # like {"key": "CICS 109: ...", "value": [...]}
    loader = JSONLoader(file_path=file_path, jq_schema="to_entries[]", text_content=False)
    documents = loader.load()

    # Convert structured JSON values (dicts/lists) into readable text that
    # includes the keys so downstream code that expects a string page_content
    # will work. Preserve the original structured value in metadata['raw_json'].
    for doc in documents:
        content = getattr(doc, "page_content", None)
        # If the loader returned a structured value, stringify it with keys
        # and keep the original structured value in metadata['raw_json'].
        if isinstance(content, dict):
            doc.metadata = dict(doc.metadata or {})
            doc.metadata["raw_json"] = content

            # Special-case entries produced by `to_entries[]`: they have
            # {'key': <key>, 'value': <value>}. We want page_content to start
            # with the key (the course title) followed by the value lines.
            if "key" in content and "value" in content:
                key = content.get("key")
                val = content.get("value")

                def fmt(v):
                    if isinstance(v, (dict, list)):
                        return json.dumps(v, ensure_ascii=False)
                    return str(v)

                if isinstance(val, list):
                    lines = [fmt(x) for x in val]
                    # Prepend the key as the first line
                    doc.page_content = str(key) + "\n" + "\n".join(lines)
                else:
                    # value is a scalar or object
                    doc.page_content = str(key) + "\n" + fmt(val)
            else:
                # Generic dict -> produce key: value lines
                lines = [f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in content.items()]
                doc.page_content = "\n".join(lines)
        elif isinstance(content, list):
            doc.metadata = dict(doc.metadata or {})
            doc.metadata["raw_json"] = content
            # join list elements, JSON-encoding each element where needed
            def fmt(v):
                if isinstance(v, (dict, list)):
                    return json.dumps(v, ensure_ascii=False)
                return str(v)

            doc.page_content = "\n".join(fmt(v) for v in content)

    return documents

if __name__ == "__main__":
    input_dir = "data/processed/json"
    documents=[]
    # `load_json` already returns a list of Document objects for each file.
    # Use extend so we create a flat list of Documents instead of a list of lists.
    for input_file in Path(input_dir).glob("*.json"):
        documents.extend(load_json(str(input_file)))
    print(f"Loaded {len(documents)} documents from {input_dir}")
    with open("test/json_loader_output.txt", "w") as f:
        f.write(str(documents))