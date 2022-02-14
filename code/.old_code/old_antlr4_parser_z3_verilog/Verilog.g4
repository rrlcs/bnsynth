grammar Verilog;

source_text
   : timing_spec? description* EOF
   ;

description
   : module_declaration
   ;

module_declaration
   : attribute_instance* module_keyword module_identifier (module_parameter_port_list)? (list_of_ports)? ';' module_item* 'endmodule'
   | attribute_instance* module_keyword module_identifier (module_parameter_port_list)? (list_of_port_declarations)? ';' non_port_module_item* 'endmodule'
   ;

module_keyword
   : 'module'
   | 'macromodule'
   ;

module_identifier
   : identifier
   ;

parameter_declaration
   : parameter_declaration_ ';'
   ;
parameter_declaration_
   : 'parameter' ('signed')? (range_)? list_of_param_assignments
   | 'parameter' 'integer' list_of_param_assignments
   | 'parameter' 'real' list_of_param_assignments
   | 'parameter' 'realtime' list_of_param_assignments
   | 'parameter' 'time' list_of_param_assignments
   ;

module_parameter_port_list
   : '#' '(' parameter_declaration_ (',' parameter_declaration_)* ')'
   ;

list_of_ports
   : '(' port (',' port)* ')'
   ;

list_of_port_declarations
   : '(' port_declaration (',' port_declaration)* ')'
   | '(' ')'
   ;

port
   : port_expression?
   | '.' port_identifier '(' (port_expression)? ')'
   ;

port_expression
   : port_reference
   | '{' port_reference (',' port_reference)* '}'
   ;

port_reference
   : port_identifier
   | port_identifier '[' constant_expression ']'
   | port_identifier '[' range_expression ']'
   ;

port_declaration
   : attribute_instance* inout_declaration
   | attribute_instance* input_declaration
   | attribute_instance* output_declaration
   ;

expression
   : term (binary_operator attribute_instance* term | '?' attribute_instance* expression ':' term)*
   ;

term
   : unary_operator attribute_instance* primary
   | primary
   | String
   ;

unary_operator
   : '+'
   | '-'
   | '!'
   | '~'
   | '&'
   | '~&'
   | '|'
   | '~|'
   | '^'
   | '~^'
   | '^~'
   ;

binary_operator
   : '+'
   | '-'
   | '*'
   | '/'
   | '%'
   | '=='
   | '!='
   | '==='
   | '!=='
   | '&&'
   | '||'
   | '**'
   | '<'
   | '<='
   | '>'
   | '>='
   | '&'
   | '|'
   | '^'
   | '^~'
   | '~^'
   | '>>'
   | '<<'
   | '>>>'
   | '<<<'
   ;

attribute_instance
   : '(' '*' attr_spec (',' attr_spec)* '*' ')'
   ;

attr_spec
   : attr_name '=' constant_expression
   | attr_name
   ;

attr_name
   : identifier
   ;

constant_expression
   : expression
   ;

identifier
   : Simple_identifier
   ;

primary

   : number
   | hierarchical_identifier
   | hierarchical_identifier ('[' expression ']') +
   | function_call
   ;

String
   : '"' (~ [\n\r])* '"'
   ;

number
   : Decimal_number
   ;

Decimal_number
   : Unsigned_number
   ;

fragment Unsigned_number
   : Decimal_digit ('_' | Decimal_digit)*
   ;

fragment Decimal_digit
   : [0-9]
   ;

hierarchical_identifier
   : simple_hierarchical_identifier
   ;

simple_hierarchical_identifier
   : simple_hierarchical_branch
   ;

simple_hierarchical_branch
   : Simple_identifier
   ;

Simple_identifier
   : [a-zA-Z_] [a-zA-Z0-9_$]*
   ;

function_call
   : hierarchical_function_identifier attribute_instance* '(' (expression (',' expression)*)? ')'
   ;

hierarchical_function_identifier
   : hierarchical_identifier
   ;

White_space
   : [ \t\n\r] + -> channel (HIDDEN)
   ;

ErrorCharacter : . ;