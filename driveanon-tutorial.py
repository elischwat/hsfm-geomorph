# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# A little driveanon tutorial

import driveanon as da
folder_blob_id = '1-2lmlxdRt1XfaHQDbpfgXpLueovdRzsy'
da.list_blobs(folder_blob_id, '.csv')
