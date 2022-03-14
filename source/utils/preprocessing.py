'''

This is a module that contains utility funcitons to help 
work with data preprocessing.

'''

def gmt_to_table(file, out='dict'):

    '''
        Transforms a .gmt gene set database into a 
        primitive tabular format. 
        Interface:
            > file: filepath to read data from
            > out: desired output type. [list / dict]
        Output:
            > col-0: Functional annotation
            > col-1: Entrez gene identifier

        The gene set table is in the form 
        (functional group 1) (url) (protein) (protein) (protein) ....
        (functional group 2) (url) (protein) (protein) (protein) ....

        This function transforms that to this
        (functional group 1) (protein)
        (functional group 1) (protein)
        (functional group 1) (protein)
        (functional group 2) (protein)
        (functional group 2) (protein)
        (functional group 2) (protein)

        Useful links:
        > https://software.broadinstitute.org/cancer/software/gsea/wiki/index.php/Data_formats#GMT:_Gene_Matrix_Transposed_file_format_.28.2A.gmt.29
    '''

    try:
        
        with open(file, 'r') as file_in:

            # init cols - position 0 is header
            col_0, col_1 = ['HALLMARK'], ['IDENTIFIER']
            
            while True:

                line = file_in.readline()
                if not line:
                    break

                # split tsv's 
                tokens = line.split('\t')

                # extract the label and url
                hallmark = tokens[0]

                url = tokens[1]

                # create the entries - start at 2 to skip url
                for idx in range(2, len(tokens)):

                    gene = tokens[idx]
                    # remove any newline characters
                    if not gene.isdigit():
                        gene = gene.replace('\n', '')
                    col_0.append(hallmark)
                    col_1.append(int(gene))
            
            # return the table
            if out == 'list':
                return [col_0, col_1]
            elif out == 'dict':
                return {col_0[0]:col_0[1:], col_1[0]:col_1[1:]}
    
    except OSError as err:

        print("OS error: {0}".format(err))

    except BaseException as err:

        print(f"Unexpected {err=}, {type(err)=}")
        raise
