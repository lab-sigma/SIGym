from flask import Flask
from flask import Blueprint, current_app, g, json, render_template, request
from src_sigym import gen
bp = Blueprint("main", __name__)

@bp.route('/', methods=['POST', 'GET'])
def index():
    l_res, f_res, rgt, m, n= gen.main()
    print(l_res, f_res, rgt)
    return render_template('base.html', l_res = l_res, f_res = f_res, rgt = rgt, m = m, n = n)