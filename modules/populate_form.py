import json
import os
from pprint import pprint
from PIL import Image, ImageDraw, ImageFont
from modules import build_config
from modules.utils import prepare_df_retrieval, prepare_dict_bounding_box, prepare_dict_form57, prepare_dict_idx_mapping

DN_DATA = 'data'
FN_BOUNDING_BOX_IMG = "bounding_box_img.png"
FP_BOUNDING_BOX_IMG = os.path.join(DN_DATA, FN_BOUNDING_BOX_IMG)


def resize_bounding_box(bounding_box):
    x_multiplier = 1.7
    y_multiplier = 2.2
    # x_offset = 60
    # y_offset = 50
    
    bounding_box["x"] = bounding_box["x"] * x_multiplier #+ x_offset
    bounding_box["y"] = bounding_box["y"] * y_multiplier #+ y_offset
    bounding_box["width"] = bounding_box["width"] * x_multiplier
    bounding_box["height"] = bounding_box["height"] * y_multiplier

    return bounding_box

def draw_bounding_boxes(outline="red", line_width=2):
    form57_img = Image.open(cfg.path.form57_img).convert("RGB")
    draw = ImageDraw.Draw(form57_img)
    
    dict_bounding_box = prepare_dict_bounding_box(cfg)

    # for field_idx, field in dict_bounding_box.items():
    #     answer_places = field["answer_places"]
    #     for answer_place, bounding_box in answer_places.items():
    #         if 'x' not in bounding_box:
    #             for sub_ap, sub_box in bounding_box.items():
    #                 draw_bounding_box(draw, sub_box)
    #         else:
    #             draw_bounding_box(draw, bounding_box)
    
    for field_idx, field in dict_bounding_box.items():
        bounding_boxes = field["bounding_boxes"]
        for bounding_box in bounding_boxes:
            draw_bounding_box(draw, bounding_box, outline, line_width)

    form57_img.save(FP_BOUNDING_BOX_IMG)
    

def draw_bounding_box(draw, bounding_box, outline, line_width):
    bounding_box = resize_bounding_box(bounding_box)
    x = bounding_box["x"]
    y = bounding_box["y"]
    width = bounding_box["width"]
    height = bounding_box["height"]

    draw.rectangle([x, y, x + width, y + height], outline=outline, width=line_width)


def populate_fields(cfg, sr_retrieved_info):
    form57_img = Image.open(cfg.path.form57_img).convert("RGB")
    draw = ImageDraw.Draw(form57_img)

    dict_bounding_box = prepare_dict_bounding_box(cfg)
    dict_idx_mapping, dict_idx_mapping_inverse = prepare_dict_idx_mapping(cfg)

    for col_idx_form, col_idx_json in dict_idx_mapping.items():
        if col_idx_json == '':
            continue
        retrieved_info = sr_retrieved_info[col_idx_json]
        if retrieved_info == 'Unknown':
            continue
        if col_idx_form == '5_month':
            retrieved_info = f'{int(retrieved_info):02d}'
            retrieved_info = f'{retrieved_info[0]} {retrieved_info[1]}'
        elif col_idx_form == '5_day':
            retrieved_info = f'{int(retrieved_info):02d}'
            retrieved_info = f'{retrieved_info[0]} {retrieved_info[1]}'
        elif col_idx_form == '5_year':
            retrieved_info = f'{retrieved_info[:2]} {retrieved_info[2:]}'

        bounding_boxes = dict_bounding_box[col_idx_form]["bounding_boxes"]
        assert len(bounding_boxes) != 0
        if len(bounding_boxes) == 1:
            bounding_box = resize_bounding_box(bounding_boxes[0])
            text = str(retrieved_info)
        elif len(bounding_boxes) == 2:
            if col_idx_form in ['6_ampm', '12_ownership']:
                if retrieved_info.lower() in ['am', 'public']:
                    bounding_box = resize_bounding_box(bounding_boxes[0])
                elif retrieved_info.lower() in ['pm', 'private']:
                    bounding_box = resize_bounding_box(bounding_boxes[1])
                else:
                    raise ValueError(f"Unexpected retrieved_info for AM/PM value: {retrieved_info}")
            else:
                raise NotImplementedError(f"Multiple bounding boxes for one field is not implemented yet for column: {col_idx_form}")
            text = 'X'
        else:
            raise NotImplementedError("Multiple bounding boxes for one field is not implemented yet.")

        position = (bounding_box["x"], bounding_box["y"])
        color = (0, 50, 360)      # black
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            size=25   # 👈 change this number
        )
        draw.text(position, text, fill=color, font=font)

    # form57_img.save(FP_BOUNDING_BOX_IMG)
    return draw


if __name__ == "__main__":
    args = {
        "c_api": "Google",
        "c_model": "gemini-2.5-flash",
        "c_n_generate": 4,
        "c_json_source": "img",
        "c_seed": 1,
        "r_api": "Huggingface",
        "r_model": "microsoft/phi-4",
        "r_n_generate": 1,
        "r_question_batch": "group"
    }
    cfg = build_config(args)

    draw_bounding_boxes(outline="red", line_width=2)
    
    df_retrieval = prepare_df_retrieval(cfg)
    idx_content = df_retrieval.columns.get_loc('content')
    sr_retrieved_info = df_retrieval.iloc[0, idx_content + 1:] # type: ignore
    populate_fields(cfg, sr_retrieved_info)