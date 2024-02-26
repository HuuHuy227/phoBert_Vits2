from text import vietnamese, cleaned_text_to_sequence

def clean_text(text):
    norm_text = vietnamese.text_normalize(text) # This can not be normilized because g2p can normalize (norm_text = vietnamese.text_normalize(text))
    phones, tones, word2ph = vietnamese.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text):
    norm_text = text
    phones, tones, word2ph = vietnamese.g2p(norm_text)
    bert = vietnamese.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert


def text_to_sequence(text):
    norm_text, phones, tones, word2ph = clean_text(text)
    return cleaned_text_to_sequence(phones, tones)


if __name__ == "__main__":
    pass
