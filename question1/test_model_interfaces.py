"""Optional offline API smoke test using tiny RANDOM weights, not real inference.

Run after installing dependencies: python -m unittest -v test_model_interfaces
"""
import string
import unittest
from types import SimpleNamespace
import numpy as np


class ModelInterfaceTests(unittest.TestCase):
    def test_fast_roberta_chunks_and_word_offsets(self):
        import torch
        from tokenizers import ByteLevelBPETokenizer
        from transformers import RobertaTokenizerFast, RobertaConfig, RobertaForSequenceClassification
        from extractors import Extractors
        backend = ByteLevelBPETokenizer()
        backend.train_from_iterator(['Good mood!'], vocab_size=261, min_frequency=1000,
                                    special_tokens=['<s>','<pad>','</s>','<unk>','<mask>'])
        tokenizer = RobertaTokenizerFast(tokenizer_object=backend._tokenizer,
                                         bos_token='<s>',eos_token='</s>',unk_token='<unk>',
                                         sep_token='</s>',cls_token='<s>',pad_token='<pad>',mask_token='<mask>')
        config = RobertaConfig(vocab_size=len(tokenizer),hidden_size=16,num_hidden_layers=1,
                               num_attention_heads=2,intermediate_size=32,num_labels=7)
        vocab = {c:j for j,c in enumerate(['<pad>','|',"'"]+list(string.ascii_uppercase))}
        class AudioProcessor:
            tokenizer = SimpleNamespace(get_vocab=lambda:vocab)
            def __call__(self, signal, **kwargs):
                return SimpleNamespace(input_values=torch.tensor(signal).unsqueeze(0))
        class AcousticModel:
            config = SimpleNamespace(pad_token_id=0,conv_kernel=[400],conv_stride=[320])
            def __call__(self, inputs):
                return SimpleNamespace(logits=torch.zeros(1,1200,len(vocab)))
        engines = Extractors.__new__(Extractors)
        engines.torch,engines.device = torch,'cpu'
        engines.tp,engines.tm = tokenizer,RobertaForSequenceClassification(config).eval()
        engines.ap,engines.am = AudioProcessor(),AcousticModel()
        engines.text_dim = 23
        text = 'Good ' * 100
        words,values,intervals,mask,chunks,error = engines.text(np.zeros(24*16000,np.float32),
                              np.ones(24*16000,bool),text,24.,0.)
        self.assertEqual(len(words),100)
        self.assertGreater(len(chunks),1)
        self.assertEqual(values.shape,(100,23))
        self.assertTrue(mask.all())
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(values[:,-7:].sum(1),1,atol=1e-6)
        self.assertTrue((intervals[:,1]>intervals[:,0]).all())
        self.assertIsNone(error)
        words2, values2, intervals2, mask2, chunks2, error2 = engines.text(
            np.zeros(24*16000, np.float32), np.zeros(24*16000, bool), text, 24., 0.)
        self.assertEqual(error2, 'no_audio_observed')
        self.assertEqual(values2.shape, (100, 23))
        self.assertTrue(np.isfinite(values2).all() and np.any(values2))
        self.assertFalse(mask2.any())
        self.assertTrue((intervals2 == 0).all())
        empty = engines.text(np.zeros(10, np.float32), np.ones(10, bool), '...', 1., 0.)
        self.assertEqual(empty[-1], 'empty_transcript')
        self.assertEqual(empty[1].shape, (0, 23))


if __name__ == '__main__':
    unittest.main()
