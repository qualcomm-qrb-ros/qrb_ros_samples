
import json
from collections import defaultdict


def build_image_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # category_id -> category_name
    category_id_to_name = {
        c["id"]: c["name"]
        for c in data.get("categories", [])
    }

    # attribute_id -> attribute_name
    attribute_id_to_name = {
        a["id"]: a["name"]
        for a in data.get("attributes", [])
    }

    # image_id -> file_name
    image_id_to_filename = {
        img["id"]: img["file_name"]
        for img in data.get("images", [])
    }

    # image_id -> collected data
    image_categories = defaultdict(set)
    image_attributes = defaultdict(set)

    for ann in data.get("annotations", []):
        image_id = ann["image_id"]

        # category
        cat_id = ann["category_id"]
        cat_name = category_id_to_name.get(cat_id)
        if cat_name is not None:
            image_categories[image_id].add(cat_name)

        # attributes
        for attr_id in ann.get("attribute_ids", []):
            attr_name = attribute_id_to_name.get(attr_id)
            if attr_name is not None:
                image_attributes[image_id].add(attr_name)

    # assemble final result
    result = {}
    for image_id in image_categories.keys() | image_attributes.keys():
        file_name = image_id_to_filename.get(image_id)
        if file_name is None:
            continue

        file_name = "./test/" + file_name

        categories = sorted(image_categories[image_id])
        attributes = sorted(image_attributes[image_id])

        metadata = concat_categories_and_attributes(categories, attributes)

        result[file_name] = {
            "metadata": metadata
        }

    return result



def concat_categories_and_attributes(categories, attributes):
    """
    Directly concatenate categories and attributes into ONE sentence.
    No inference, no reordering magic, no NLP.
    """
    parts = []

    if categories:
        parts.extend(categories)

    if attributes:
        parts.extend(attributes)

    return ", ".join(parts)


def save_to_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    input_json = "./instances_attributes_val2020.json"
    output_json = "./file_name_and_metadata.json"

    dataset = build_image_dataset(input_json)
    save_to_json(dataset, output_json)

    print(f"Saved {len(dataset)} samples to {output_json}")
