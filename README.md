

# Environment Setup

Our algorithms are implemented in Python 3.7.3  and  all experiments are conducted on a Linux server with one Intel(R) Xeon(R) Silver 4210 2.20GHz CPU and 1 TB RAM.

Ps: Our code can also run on a desktop with Apple M1 and 8GB RAM running macOS Monterey 12.3  (and Inter(R) Core(TM) i7-10700@2.90GHZ and 16 GB RAM running Windows 10). But we recommend you run it on a server because the server has enough memory to handle large datasets and run faster.




# Dataset

Due to the space limit, we do not upload Youtube, LiveJ, and Orkut. But, you can download all original datasets used in our paper from http://snap.stanford.edu/

We also provide generator.py to generate the synthetic graphs used in our paper. Please see https://networkx.org/ for more details.

If you have any questions, please contact longlonglin@swu.edu.cn


 # Usage
   You may use git to clone the repository from GitHub and run it manually like this
  
      git clone https://github.com/longlonglin/PSMC.git
      cd PSMC
      python PSMC.py com-amazon.ungraph.txt

   The  running results are as follows
   
    ........................
    com-amazon.ungraph.txt is loading...
    load_time(s)2.1507396697998047
    number of nodes334863
    number of edges925872.0
    core_decomposition_time(s)5.175716161727905
    degeneracy6
    clique_number667129.0
    max_cc  189145
    the time of compute the weighted graph(s)12.37115740776062
    PSMC_time9.245013952255249
    PSMC_conductance0.000453755724187932
    PSMCplus_time8.347105979919434
    PSMCplus_conductance0.012952027929830611
    avg_MAPPR_time16.912541580200195
    avg_MAPPR_condu0.014968469376126564
    avg_HOSPLOC_time12.395989227294923
    avg_HOSPLOC_condu0.10506984344394185
    HSC_time337.5114724636078
    HSC_conductance0.06796116504854369
    SC_time26.020264387130737
    SC_conductance0.7042086656397121
    louvain_time70.71251249313354
    louvain_conductance0.006705615388724329
    KCore_time5.315958023071289
    KCore_conductance0.10929994487754942
    HD_time22.09650206565857
    HD_conductance0.2698648781189078
    HM_time36.828208923339844
    HM_conductance0.007004705870680733

   Our model has only one parameter, ``k``, which ranges from 3 to 6, and its default value is 3. If you want to change ``k``, you can modify it in Line 824 of PSMC.py.
   
   
