import unittest
from unittest.mock import patch, MagicMock
from src.utils import create_embeddings_batch
from typing import List

class TestCreateEmbeddingsBatch(unittest.TestCase):
    
    def test_empty_input_returns_empty_list(self):
        """
        TC01: 输入为空列表时，应直接返回空列表
        """
        result = create_embeddings_batch([])
        self.assertEqual(result, [])
    
    @patch('utils.vikingdb_service')
    @patch('utils.vikingdb_service.embedding_v2')
    def test_successful_embedding_creation(self, mock_openai_create, mock_vikingdb):
        """
        TC02: OpenAI 返回有效 embedding，应正确解析并返回
        """
        # 构造模拟返回值
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[-0.1, -0.2, -0.3])
        ]
        mock_openai_create.return_value = mock_response
        
        texts = ["hello", "world"]
        result = create_embeddings_batch(texts)
        
        expected_result = [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]
        self.assertEqual(result, expected_result)
        mock_openai_create.assert_called_once_with(
            model="text-embedding-3-small",
            input=texts
        )
    
    @patch('utils.vikingdb_service')
    @patch('utils.vikingdb_service.embedding_v2', side_effect=Exception("API Error"))
    def test_exception_handling_returns_zero_vectors(self, mock_openai_create, mock_vikingdb):
        """
        TC03: OpenAI 抛出异常，应返回与输入等长的零向量（维度1536）
        """
        texts = ["hello", "world"]
        result = create_embeddings_batch(texts)
        
        # 每个嵌入向量长度为 1536
        self.assertEqual(len(result), len(texts))
        for vec in result:
            self.assertEqual(len(vec), 1536)
            self.assertTrue(all(x == 0.0 for x in vec))

if __name__ == '__main__':
    unittest.main()