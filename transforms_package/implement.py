
x_pipeline = Pipeline([
        ('drop_columns',DropColumns(["NOME","MATRICULA"])),
        #('custom_remove_nan',CustomRemoveNanValues(["NOTA_GO"],["PERFIL","INGLES"],False)),
        #('custom_remove_nan2',CustomRemoveNanValues(["INGLES"],["PERFIL"],True)),
        ('feature_score',MedianRow(["NOTA_DE","NOTA_MF","NOTA_GO","NOTA_EM"])),
        ('merge_two_columns1',MegeTwoColumns(["NOTA_MF","NOTA_GO"],"M_E")),
        ('merge_two_columns2',MegeTwoColumns(["NOTA_DE","NOTA_EM"],"M_H")),
        ('merge_two_columns3',MegeTwoColumns(["H_AULA_PRES","TAREFAS_ONLINE"],"ATIVIDADES")),
        ('feture_dificuldade',FeaturesDificuldade()),
        ('standar_scaler',CustomStandardScaler())
    ])