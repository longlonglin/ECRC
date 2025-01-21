

# Environment Setup

Our algorithms are implemented in Python 3.7.3  and  all experiments are conducted on a Linux server with one Intel(R) Xeon(R) Silver 4210 2.20GHz CPU and 1 TB RAM.

Ps: Our code can also run on a desktop with Apple M1 and 8GB RAM running macOS Monterey 12.3  (and Inter(R) Core(TM) i7-10700@2.90GHZ and 16 GB RAM running Windows 10). But we recommend you run it on a server because the server has enough memory to handle large datasets and run faster.




 # Usage

python ECRC.py IMDB.txt

The  running results are as follows

IMBD.txt is loading...
number of left_nodes617
number of right_nodes1398
average_left_degree8.625607779578607
average_right_degree3.8068669527896994
number of edges5322.0
p,q: 2  2
################ECRC###########
quality_ECRC  0.0028099754127151387
time of ECRC  0.2311077117919922
################ECRC_E###########
quality_ECRC_E  0.0
time of ECRC_E  0.15845584869384766
p,q: 2  3
################ECRC###########
quality_ECRC  0.0005022601707684581
time of ECRC  0.08385944366455078
################ECRC_E###########
quality_ECRC_E  0.0
time of ECRC_E  0.16689419746398926
p,q: 3  2
################ECRC###########
quality_ECRC  0.0
time of ECRC  0.18693304061889648
################ECRC_E###########
quality_ECRC_E  0.0
time of ECRC_E  0.15209364891052246
p,q: 2  4
################ECRC###########
quality_ECRC  0.007633587786259542
time of ECRC  0.017497777938842773
################ECRC_E###########
quality_ECRC_E  0.0
time of ECRC_E  0.15550017356872559
p,q: 3  3
################ECRC###########
quality_ECRC  0.0
time of ECRC  0.01649165153503418
################ECRC_E###########
quality_ECRC_E  0.0
time of ECRC_E  0.15317773818969727
p,q: 4  2
################ECRC###########
quality_ECRC  0.0
time of ECRC  0.0749671459197998
################ECRC_E###########
quality_ECRC_E  0.0
time of ECRC_E  0.1459662914276123



