# -*- coding: utf-8 -*-

'''
Claim analysis: Analyze extracted claims from the narrative blueprint.
It uses embeddings to analyze the claims, measure semantic similarity,
and detect communities.
'''

# import modules
import os
import csv
import pandas as pd
import networkx as nx

# skleanr and community detection
from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms.community import louvain_communities

# embeddings
from embeddings import VectorModel

# MongoDB connection
from databases import MongoDBManager

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
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # MongoDB connection
        self.mongodb_manager = MongoDBManager()
        self.mongo_db_name = self.args['mongo_db_name']
        self.mongo_collection_name = self.args['mongo_collection_name']

        # community analysis results
        self.threshold = self.args['threshold']
        self.analysis_results = {
            'total_communities': 0,
            'communities': []
        }

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
    
    def _build_similarity_graph(self,
                                claims: list[str],
                                embeddings: list[list[float]],
                                threshold: float = 0.75) -> nx.Graph:
        '''
        Builds a similarity graph from the given claims and embeddings.

        :param claims: The claims to be used for building the graph.
        :type claims: list[str]

        :param embeddings: The embeddings to be used for building the graph.
        :type embeddings: list[list[float]]

        :param threshold: The threshold to be used for building the graph.
        :type threshold: float

        :return: A similarity graph.
        :rtype: nx.Graph
        '''
        # compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # graph
        G = nx.Graph()
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                if similarity_matrix[i][j] >= threshold:
                    G.add_edge(
                        i,
                        j,
                        weight=similarity_matrix[i][j],
                        similarity=similarity_matrix[i][j],
                        claim_pair=(claims[i], claims[j]),
                        embedding_pair=(embeddings[i], embeddings[j])
                    )
        
        # add nodes
        for node in G.nodes:
            G.nodes[node]['claim'] = claims[node]
        
        return G
    
    def _analyze_communities(self, G: nx.Graph,
                             claims: list[str]) -> None:
        '''
        Analyzes the communities in the similarity graph.

        :param G: NetworkX graph
        :type G: nx.Graph

        :param claims: The claims to be used for the analysis.
        :type claims: list[str]
        '''
        communities = louvain_communities(G, seed=42)
        for idx, community in enumerate(communities):
            if len(community) >= 2:
                community_claims = list(set([claims[node] for node in community]))

                # get all edges
                community_edges = []
                unique_pairs = set()
                for edge in G.edges(data=True):
                    node1, node2, attrs = edge
                    if node1 in community or node2 in community:
                        pair_key = tuple(sorted((node1, node2)))
                        if pair_key not in unique_pairs:
                            unique_pairs.add(pair_key)
                            community_edges.append(
                                {
                                    'nodes': (node1, node2),
                                    'similarity': attrs['similarity'],
                                    'claims': attrs['claim_pair'],
                                    'embedding': attrs['embedding_pair']
                                }
                            )
                
                # avg similarity
                avg_similarity = sum(edge['similarity'] for edge in community_edges) \
                    / len(community_edges) if community_edges else 0

                # build community info
                community_info = {
                    'size': len(community),
                    'nodes': list(community),
                    'claims': community_claims,
                    'edges': community_edges,
                    'avg_similarity': avg_similarity
                }

                # add community info to analysis results
                self.analysis_results['total_communities'] += 1
                self.analysis_results['communities'].append(community_info)
                
        return
    
    def _export_community_analysis(self) -> None:
        '''
        Exports community analysis results to CSV files.

        :return: None
        '''
        # export detailed community claims
        detailed_data = []
        for idx, community in enumerate(self.analysis_results['communities']):
            for claim in community['claims']:
                detailed_data.append({
                    'claim': claim
                })

        # export to XLSX
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_excel(
            f'{self.output_path}/community_claims.xlsx',
            index=False
        )

        # export similarity pairs
        pairs_data = []
        for community in self.analysis_results['communities']:
            for edge in community['edges']:
                pairs_data.append({
                    'node1': edge['nodes'][0],
                    'node2': edge['nodes'][1],
                    'similarity': edge['similarity'],
                    'claim1': edge['claims'][0],
                    'claim2': edge['claims'][1]
                })

        # export to XLSX
        pairs_df = pd.DataFrame(pairs_data)
        pairs_df.to_excel(
            f'{self.output_path}/community_similarity_pairs.xlsx',
            index=False
        )

        return

    def run_similarity_analysis(self,
                                claims: list[str],
                                embeddings: list[list[float]],
                                threshold: float = 0.75) -> None:
        '''
        Runs the similarity analysis.

        :param claims: The claims to be used for the similarity analysis.
        :type claims: list[str]

        :param embeddings: The embeddings to be used for the similarity analysis.
        :type embeddings: list[list[float]]

        :param threshold: The threshold to be used for the similarity matrix.
        :type threshold: float
        '''
        # split claims data into batches
        batch_size = 1000
        batch_claims = [
            claims[i:i+batch_size]
            for i in range(0, len(claims), batch_size)
        ]

        batch_embeddings = [
            embeddings[i:i+batch_size]
            for i in range(0, len(embeddings), batch_size)
        ]

        # build similarity graph for each batch
        for claims, embeddings in zip(batch_claims, batch_embeddings):
            G = self._build_similarity_graph(
                claims=claims,
                embeddings=embeddings,
                threshold=threshold
            )

            # analyze communities
            self._analyze_communities(G, claims)
        
        # extract results from analysis
        claims_above_threshold = [
            c
            for community in self.analysis_results['communities']
            for edge in community['edges']
            for c in edge['claims']
        ]

        embeddings_above_threshold = [
            e
            for community in self.analysis_results['communities']
            for edge in community['edges']
            for e in edge['embedding']
        ]

        # unique claims and embeddings
        unique_claims = []
        unique_embeddings = []
        for idx, claim in enumerate(claims_above_threshold):
            if claim not in unique_claims:
                unique_claims.append(claim)
                unique_embeddings.append(embeddings_above_threshold[idx])
        unique_claims = list(unique_claims)

        if len(unique_claims) == 0 or len(unique_embeddings) == 0:
            print('WARNING: No claims found above threshold. Skipping final similarity analysis.')
            return

        # rebuild similarity graph
        G = self._build_similarity_graph(
            claims=unique_claims,
            embeddings=unique_embeddings,
            threshold=threshold
        )

        # analyze communities
        self.analysis_results = {
            'total_communities': 0,
            'communities': []
        }
        self._analyze_communities(G, unique_claims)

        # export community analysis
        self._export_community_analysis()
    
    def run_claim_analysis(self) -> None:
        '''
        Runs the full pipeline to embed narrative claims.

        :return: None
        '''
        claims = self.claims['claim'].tolist()
        uuids = self.claims['uuid'].tolist()

        # compute embeddings
        self.vector_model.compute_embeddings(
            uuids=uuids,
            data=claims,
            mongo_db_name=self.mongo_db_name,
            mongo_collection_name=self.mongo_collection_name
        )

        # load embeddings from MongoDB
        documents = self.mongodb_manager.get_documents(
            db_name=self.mongo_db_name,
            collection_name=self.mongo_collection_name
        )

        # writing embeddings to projector
        print ('')
        print ('Writing embeddings for Embedding Projector...')
        print ('Visit: https://projector.tensorflow.org/')
        self._embedding_projector(
            claims=[item['text'] for item in documents],
            embeddings=[item['embedding'] for item in documents]
        )
        print ('Done.')

        # run similarity analysis
        print ('')
        print ('Running similarity analysis...')
        self.run_similarity_analysis(
            claims=[item['text'] for item in documents],
            embeddings=[item['embedding'] for item in documents],
            threshold=self.threshold
        )
        print ('finished.')

        return
