# Visualizer

This tool helps visualize the results from the xclip tool.

Before running the `calc` files, make sure to run the `_prep.py` file first. This reads the data and prepares 
Files that start with `calc` are used to calculate a result from the animal kingdom data. The results are stored as a .json file in the `data/` folder.
Files that start with `draw` are used to draw the results from the `data` folder. The results are stored as a .png file in the `figures/` folder.

# Files
`_prep.py` - Prepares the data for the `calc` files.
`data.json` - Contains the data from the animal kingdom used for action recognition on videos.
`calc_xclip.py` - Uses XCLIP for AR. Prompt: "a {class} is {action}" for every combination of class and action
- -g or --granularity: Granularity of the action recognition. Options: "class", "animal", "nothing"

`draw_overall_avg.py` - Draws histograms of average accuracy for each granularity.

# File Formats

### data.json
data.json is a dictionary containing data relevant to action recognition
```
{
    "action_index_key": [
        "<action_name>",
        "<action_name>",
        ...
    ],
    "video_data": {
        "<video_path>": {
            "animal": <animal_name>,
            "animal_parent_class": <animal_parent_class>,
            "animal_class": <animal_class>,
            "animal_subclass": "<animal_subclass>",
            "action": <action_index>
        },
        ...
    }
}
```

### data/{model}.json
The data/{model}.json file contains the experiment outputs from xclip
```
{
    "<granularity>": {
        "cm_pred": [...],
        "cm_true": [...],
        "raw_data": {
            "<video_path>": {
                "animal": <animal_name>,
                "animal_parent_class": <animal_parent_class>,
                "animal_class": <animal_class>,
                "animal_subclass": <animal_subclass>,
                "pred_action": <pred_action>,
                "true_action": <true_action>
            },
            ...
        }
    }
}
```