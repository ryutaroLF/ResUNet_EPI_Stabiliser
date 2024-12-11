
BATCHSIZE = 16
NUMof1epoch = 5 #50
MAX_EPOCH = 1000 #20
CROP_SIZE = 16 # 16 -> 8 -> 4 -> 2 



train_dataset_list = [
            "additional/antinous", "additional/boardgames", "additional/dishes",   "additional/greek",
            "additional/kitchen",  "additional/medieval2",  "additional/museum",   "additional/pens",
            "additional/pillows",  "additional/platonic",   "additional/rosemary", "additional/table",
            "additional/tomb",     "additional/tower",      "additional/town",     "additional/vinyl" ]

val_dataset_list = [
            "stratified/backgammon", "stratified/dots", "stratified/pyramids", "stratified/stripes",
            "training/boxes", "training/cotton", "training/dino", "training/sideboard"]
