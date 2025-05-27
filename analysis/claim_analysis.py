# -*- coding: utf-8 -*-

'''
Claim analysis: Analyze extracted claims from the narrative blueprint.
It uses embeddings to analyze the claims, measure semantic similarity,
and detect communities.
'''

# import modules
import csv
import pandas as pd

# embeddings
from embeddings import VectorModel

# Claim Analysis class
class ClaimAnalysis:
    '''
    ClaimAnalysis class
    '''
    def __init__(self, vector_model: VectorModel, args: dict = None):
        '''
        Initializes the ClaimAnalysis pipeline, validating input arguments
        and preparing the dataset for embedding analysis. Ensures that
        the vector model is compatible, the claims file is present and
        correctly formatted, and that required columns exist for processing.

        :param vector_model: The vector model to be used for claim analysis.
        :type vector_model: VectorModel

        :raises TypeError: If the provided `vector_model` is not a subclass
            of `VectorModel`.

        :param args: Additional arguments to be passed to the ClaimAnalysis class.
        :type args: dict

        Example:
            >>> from embeddings import OpenAIEmbeddingModel
            >>> vector_model = OpenAIEmbeddingModel(model='text-embedding-3-large')
            >>> claim_analysis = ClaimAnalysis(vector_model=vector_model)
        '''
        # get arguments
        args = args or {}
        self.args = args

        # vector model
        if not isinstance(vector_model, VectorModel):
            raise TypeError('vector_model must be an instance of VectorModel')
        
        self.vector_model = vector_model

        # load claims
        claims_path = self.args.get('claims_path')
        if not claims_path:
            raise ValueError('The `claims_path` argument is required')
        
        self.claims = self._load_claims(claims_path)

        # check required columns in claims dataset
        required_columns = ['claim', 'uuid']
        missing_columns = [
            col for col in required_columns
            if col not in self.claims.columns
        ]
        if missing_columns:
            raise ValueError(
                f'Claims dataset missing required columns: {missing_columns}'
            )
        
        # output path
        self.output_path = self.args['output']

    def _load_claims(self, path: str) -> pd.DataFrame:
        '''
        Loads a CSV file containing narrative claims into a
        pandas DataFrame.

        :param path: The path to the CSV file containing the claims.
        :type path: str

        :return: A pandas DataFrame containing the claims.
        :rtype: pd.DataFrame
        '''
        try:
            return pd.read_csv(path, encoding='utf-8')
        except Exception as e:
            raise e
    
    def _save_embeddings_results(self,
                                 uuids: list[str],
                                 claims: list[str],
                                 embeddings: list[list[float]]) -> None:
        '''
        Saves the claim UUIDs, text, and their corresponding vector
        embeddings to disk.

        Embeddings are stored alongside their original text and identifier
        to support downstream analysis such as similarity clustering or
        visualization.

        :param uuids: The UUIDs of the claims.
        :type uuids: list[str]
        
        :param claims: The claims to be saved.
        :type claims: list[str]

        :param embeddings: The embeddings of the claims.
        :type embeddings: list[list[float]]

        :return: None
        '''
        results = {
            'uuid': uuids,
            'claim': claims,
            'embeddings': embeddings 
        }

        # build dataframe
        df = pd.DataFrame(results)

        # save data
        df.to_csv(
            f'{self.output_path}/embeddings_gemini.csv',
            encoding='utf-8',
            index=False
        )
    
    def _embedding_projector(self, claims: list[str],
                             embeddings: list[list[float]]) -> None:
        '''
        Project embeddings into TensorFlow's embedding projector.

        :param claims: The claims to be projected.
        :type claims: list[str]

        :param embeddings: The embeddings to be projected.
        :type embeddings: list[list[float]]

        :return: None
        '''
        if len(claims) != len(embeddings):
            raise ValueError('The number of claims and embeddings must be the same')
        
        # write embeddings metadata file
        with open(f'{self.output_path}/_tensor.tsv', 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            for embedding in embeddings:
                writer.writerow(embedding)
        
        # write claims metadata file
        with open(f'{self.output_path}/_metadata.tsv', 'w', encoding='utf-8') as f:
            for claim in claims:
                f.write(claim + '\n')

    def run_claim_analysis(self) -> None:
        '''
        Runs the full pipeline to embed narrative claims.

        :return: None
        '''
        claims = self.claims['claim'].tolist()
        uuids = self.claims['uuid'].tolist()

        # compute embeddings
        embeddings = self.vector_model.compute_embeddings(claims)

        # process results
        self._save_embeddings_results(
            uuids=uuids,
            claims=claims,
            embeddings=embeddings
        )

        # writing embeddings to projector
        print ('')
        print ('Writing embeddings to projector...')
        self._embedding_projector(
            claims=claims,
            embeddings=embeddings
        )
        print ('finished.')
