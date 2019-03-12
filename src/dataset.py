import numpy as np
import os
import xml.etree.ElementTree as etree
import pprint as pp


def mark_pos(x, y, z, row, pos):
    x[row, pos] = 1
    y[row, pos] = 1
    if row > 0:
        z[row - 1, pos] = 1


def read_data(data_dir):
    pos_types, max_len = build_pos_dictionary(data_dir)
    pos_types['<start>'] = len(pos_types)
    pos_types['<mark>'] = len(pos_types)
    pos_types['<end>'] = len(pos_types)

    embedding_size = len(pos_types) + 1
    pp.pprint(pos_types)
    print("Embedding size: {}.".format(embedding_size))
    print("Max length: {}.".format(max_len))
    x, y, z = [], [], []
    for f in enumerate_files(data_dir):
        for upper_seg in f.findall('seg'):
            tok_x = np.zeros((max_len, embedding_size))
            tok_y = np.zeros_like(tok_x)
            tok_z = np.zeros_like(tok_x)
            x.append(tok_x)
            y.append(tok_y)
            z.append(tok_z)

            # Mark start of sequence
            mark_pos(tok_x, tok_y, tok_z, 0, pos_types['<start>'])
            i = 1
            for child in upper_seg.iter():
                if child.tag == 'TOK' and 'pv' in child.attrib:
                    pos = child.attrib['pv']
                    mark_pos(tok_x, tok_y, tok_z, i, pos_types[pos])
                    i += 1
                if child.tag == 'cue':
                    pos = '<mark>'
                    mark_pos(tok_x, tok_y, tok_z, i, pos_types[pos])
                    i += 1
            # Mark end of sequence
            mark_pos(tok_x, tok_y, tok_z, i, pos_types['<end>'])

    x = np.reshape(x, (len(x), max_len, embedding_size))
    y = np.reshape(y, (len(y), max_len, embedding_size))
    z = np.reshape(y, (len(z), max_len, embedding_size))

    return x, y, z, pos_types


def enumerate_files(data_dir):
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                tree = etree.parse(file_path)
                yield tree.getroot()
            except UnicodeDecodeError:
                continue


def build_pos_dictionary(data_dir):
    pos_types = set()
    max_seq_length = 0
    for f in enumerate_files(data_dir):
        for tok in f.iter('TOK'):
            if 'pv' in tok.attrib:
                pos_types.add(tok.attrib['pv'])
        for seg in f.iter('seg'):
            max_seq_length = max(len(list(seg)), max_seq_length)
    return {t: i for i, t in enumerate(pos_types)}, max_seq_length


# build_pos_dictionary('/home/petru/Downloads/1984')
# x, y, z = read_data('/home/petru/Downloads/1984')
