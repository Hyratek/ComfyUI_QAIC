<div align="center">

<img src="https://github.com/user-attachments/assets/1a1d2f65-b00c-4c96-92bb-a565b4c2937f" alt="Logo HYRA TEK-01" width="300" />

## ComfyUI_QAIC



## Overview


**ComfyUI_QAIC extension designed for deploying and running diffusion models on Qualcomm Cloud AI 100 accelerators.**

(developed by Hyratek & Qualcomm)

Comfy_QAIC extends ComfyUI's powerful node-based interface to support model deployment on Qualcomm Cloud AI 100. This extension provides optimized nodes and tools for efficient model execution on QAIC hardware

This repository is a modified version of ComfyUI, adapted specifically for deployment on Qualcomm Cloud AI 100 accelerators.
</div>




## Shortcuts

| Keybind                            | Explanation                                                                                                        |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| Ctrl + Enter                       | Queue up current graph for generation                                                                              |
| Ctrl + Shift + Enter               | Queue up current graph as first for generation                                                                     |
| Ctrl + Alt + Enter                 | Cancel current generation                                                                                          |
| Ctrl + Z/Ctrl + Y                  | Undo/Redo                                                                                                          |
| Ctrl + S                           | Save workflow                                                                                                      |
| Ctrl + O                           | Load workflow                                                                                                      |
| Ctrl + A                           | Select all nodes                                                                                                   |
| Alt + C                            | Collapse/uncollapse selected nodes                                                                                 |
| Ctrl + M                           | Mute/unmute selected nodes                                                                                         |
| Ctrl + B                           | Bypass selected nodes (acts like the node was removed from the graph and the wires reconnected through)            |
| Delete/Backspace                   | Delete selected nodes                                                                                              |
| Ctrl + Backspace                   | Delete the current graph                                                                                           |
| Space                              | Move the canvas around when held and moving the cursor                                                             |
| Ctrl/Shift + Click                 | Add clicked node to selection                                                                                      |
| Ctrl + C/Ctrl + V                  | Copy and paste selected nodes (without maintaining connections to outputs of unselected nodes)                     |
| Ctrl + C/Ctrl + Shift + V          | Copy and paste selected nodes (maintaining connections from outputs of unselected nodes to inputs of pasted nodes) |
| Shift + Drag                       | Move multiple selected nodes at the same time                                                                      |
| Ctrl + D                           | Load default graph                                                                                                 |
| Alt + `+`                          | Canvas Zoom in                                                                                                     |
| Alt + `-`                          | Canvas Zoom out                                                                                                    |
| Ctrl + Shift + LMB + Vertical drag | Canvas Zoom in/out                                                                                                 |
| P                                  | Pin/Unpin selected nodes                                                                                           |
| Ctrl + G                           | Group selected nodes                                                                                               |
| Q                                  | Toggle visibility of the queue                                                                                     |
| H                                  | Toggle visibility of history                                                                                       |
| R                                  | Refresh graph                                                                                                      |
| Double-Click LMB                   | Open node quick search palette                                                                                     |
| Shift + Drag                       | Move multiple wires at once                                                                                        |
| Ctrl + Alt + LMB                   | Disconnect all wires from clicked slot                                                                             |


## Prerequisites

All Dependencies have been tested on an Ubuntu 22.04 system.

## Setup Environment

```bash
# Create Python virtual env and activate it. (Recommended Python 3.10)
sudo apt install python3.10-venv
python3.10 -m venv comfy_env
source comfy_env/bin/activate

# Install requirements
pip install -r requirements.txt

#Install the package "qaic"
pip install /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl

``` 

## Running

```bash
# Run the GUI
python main.py
```

# Support 
Use [GitHub Issues](https://github.com/Hyratek/ComfyUI_QAIC/issues) to request for model support, raise questions or to provide feedback.  

# ComfyUI_QAIC
