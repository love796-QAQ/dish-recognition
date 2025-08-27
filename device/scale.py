from recognize import recognize

last_weight = 0
records = []

def add_dish(img_path, current_weight):
    global last_weight
    dish, score = recognize(img_path)
    dish_weight = current_weight - last_weight
    last_weight = current_weight
    record = {"dish": dish, "weight": dish_weight, "similarity": round(score,3)}
    records.append(record)
    return record
