from flask import Flask
from flask import Blueprint, current_app, g, json, render_template, request
#from src_sigym import gen
bp = Blueprint("main", __name__)

l_res = [0, 1, 2, 2, 1, 0, 1, 2, 1, 0]
f_res = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0]

rgt_res = [0.4861321016343646, 0.9632937032687292, 1.4138241049030937, 1.8553840065374583, 2.270312708171823, 2.676270909806188, 3.0286642114405526, 3.3544263130749172, 3.6535572147092816, 3.9437176163436463]

rgt = 3.9437176163436463

@bp.route('/', methods=['POST', 'GET'])
def index():
    '''
    l_res, f_res, rgt_res, rgt, m, n= gen.main()
    print(l_res, f_res, rgt)
    return render_template('base.html', l_res = l_res, f_res = f_res, rgt = rgt, m = m, n = n)
    '''
    return render_template('base.html', l_res = l_res, f_res = f_res, rgt_res = rgt_res, rgt = rgt, m = 3, n = 3)