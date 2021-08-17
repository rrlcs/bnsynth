# Solution provided by @Ajax1234 on Stack Overflow
# https://stackoverflow.com/questions/68760031/replace-references-of-variables-with-their-assignments-done-later-in-python-file/ 
# Declarative to Imperative Programming language

import ast, itertools, collections as cl
import astunparse

class AssgnCheck:
   def __init__(self, scopes = None):
      self.scopes, self.missing = scopes or cl.defaultdict(lambda :cl.defaultdict(list)), []
   @classmethod
   def eq_ast(cls, a1, a2):
      #check that two `ast`s are the same
      if type(a1) != type(a2):
         return False
      if isinstance(a1, list):
         return all(cls.eq_ast(*i) for i in itertools.zip_longest(a1, a2))
      if not isinstance(a1, ast.AST):
         return a1 == a2
      return all(cls.eq_ast(getattr(a1, i, None), getattr(a2, i, None)) 
                 for i in set(a1._fields)|set(a2._fields) if i != 'ctx')
   def has_bindings(self, t_ast, s_path):
      #traverse the scope stack and yield `ast`s from t_ast that do not have a value assigned to them
      for _ast in t_ast:
         if not any(any(AssgnCheck.eq_ast(_ast, b) for _, b in self.scopes[sid]['names']) for sid in s_path[::-1]):
            yield _ast
   def traverse(self, _ast, s_path = [1]):
      #walk the ast object itself
      _t_ast = None
      if isinstance(_ast, ast.Assign): #if assignment statement, add ast object to current scope
         self.traverse(_ast.targets[0], s_path)
         self.scopes[s_path[-1]]['names'].append((True, _ast.targets[0]))
         self.scopes[s_path[-1]]['bindings'].append((_ast.targets[0], _ast.value))
         _ast = _ast.value
      if isinstance(_ast, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
         if not self.scopes:
             nid = 1
         else:
             nid = max(self.scopes)+1
         s_path = [*s_path, nid]
         if isinstance(_ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.scopes[nid]['names'].extend([(False, ast.Name(i.arg)) for i in _ast.args.args])
            _t_ast = [*_ast.args.defaults, *_ast.body]
      self.missing.extend(list(self.has_bindings(_t_ast if _t_ast is not None else [_ast], s_path))) #determine if current ast object instance has a value assigned to it
      if _t_ast is None:
         _ast.s_path = s_path
         for _b in _ast._fields:
            b = getattr(_ast, _b)
            if isinstance((b), list):
               for i in b:
                  self.traverse(i, s_path)
            elif isinstance(b, ast.AST):
               self.traverse(b, s_path)
      else:
          for _ast in _t_ast:
             _ast.s_path = s_path
             self.traverse(_ast, s_path)

import copy
def replace_vars(_ast, c_obj, sentinel):
   def ast_bindings(a, n, v, is_l = False):
      if not isinstance(v, ast.AST):
         return
      if v in c_obj.missing:
         c_obj.missing.remove(v)
         for sid in v.s_path[::-1]:
            k=[y for x, y in c_obj.scopes[sid]['bindings'] if AssgnCheck.eq_ast(v, x)]
            if (k):
               sentinel.f = True
               if not is_l:
                  setattr(a, n, copy.deepcopy(k[0]))
               else:
                  a[n] = copy.deepcopy(k[0])
               return
      replace_vars(v, c_obj, sentinel)
   if isinstance(_ast, ast.Assign):
      ast_bindings(_ast, 'value', _ast.value)
   else:
      for i in _ast._fields:
         k=getattr(_ast, i)
         if isinstance((k), list):
            for x, y in enumerate(k):
               ast_bindings(k, x, y, True)
         else:
            ast_bindings(_ast, i, k)


class Sentinel:
   def __init__(self):
      self.f = False

def replace_preref(s):
   t = ast.parse(s)
   while True:
      a = AssgnCheck()
      a.traverse(t)
      s = Sentinel()
      replace_vars(t, a, s)
      if not s.f:
         break
   return astunparse.unparse(t)
