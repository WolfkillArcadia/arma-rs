#![recursion_limit = "256"]
#![forbid(clippy::missing_docs_in_private_items)]

//! Create Arma extensions easily in Rust and the power of code generation

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Mutex;

extern crate proc_macro;
#[macro_use]
extern crate lazy_static;

use proc_macro::TokenStream;
use quote::quote;
use regex::Regex;
use syn::ItemFn;

lazy_static! {
    static ref PROXIES: Mutex<Vec<String>> = Mutex::new(Vec::new());
    static ref PROXIES_ARG: Mutex<Vec<String>> = Mutex::new(Vec::new());
}

#[proc_macro_attribute]
/// Create an RV function for use with callExtension.
///
/// # Example
///
/// ```ignore
/// use arma_rs::{rv, rv_handler};
///
/// #[rv]
/// fn hello() -> &'static str {
///    "Hello from Rust!"
/// }
///
/// #[rv]
/// fn is_arma3(version: u8) -> bool {
///     version == 3
/// }
///
/// #[rv]
/// fn say_hello(name: String) -> String {
///     format!("Hello {}", name)
/// }
///
/// #[rv(thread=true)]
/// fn do_something() {}
///
/// #[rv_handler]
/// fn init() {}
/// ```
///
/// `"myExtension" callExtension ["say_hello", ["Rust"]]` => `Hello Rust`
///
/// Any type that implements the trait [`FromStr`] can be used as an argument.
/// Any type that implements the trait [`ToStr`] can be used as the return type.
///
/// # Parameters
///
/// **Thread**
/// A function can be ran in it's own thread as long as it does not have a return value
///
/// [`FromStr`]: https://doc.rust-lang.org/std/str/trait.FromStr.html
/// [`ToStr`]: https://doc.rust-lang.org/std/string/trait.ToString.html
pub fn rv(attr: TokenStream, item: TokenStream) -> TokenStream {
    let ast = syn::parse_macro_input!(item as ItemFn);

    // Track if the current function wants to be threaded or not
    let mut is_threaded = false;
    let attribute_string = attr.to_string();

    // Determine if the current function wants to be a thread
    if !attribute_string.is_empty() {
        let regex = Regex::new(
            r#"(?m)(?P<key>[^,]+?)(?:\s+)?=(?:\s+)?(?P<value>[^",]+|"(?:[^"\\]|\\.)*")"#,
        )
        .expect("Failed to create regex");

        for captures in regex.captures_iter(&attribute_string) {
            if &captures["key"] != "thread" {
                continue;
            }

            is_threaded = bool::from_str(&captures["value"]).unwrap();
            break; // Don't continue to process
        }
    }

    // Build the _handler and _info functions names
    let name = &ast.sig.ident;
    let name_string = ast.sig.ident.to_string();
    let handler = syn::Ident::new(&format!("{}_handler", name), name.span());
    let info = syn::Ident::new(&format!("{}_info", name), name.span());

    // Create the arguments for the new functions
    let mut handler_arguments: HashMap<syn::Ident, Box<syn::Type>> = HashMap::new();
    let mut handler_argument_types: Vec<Box<syn::Type>> = Vec::new();

    // Using the arguments from the bounded function, build a new argument tree if there are any
    let rv_function_arguments = &ast.sig.inputs;
    rv_function_arguments.pairs().for_each(|p| {
        let v = p.value();
        match v {
            syn::FnArg::Typed(t) => {
                if let syn::Pat::Ident(i) = &*t.pat {
                    handler_arguments.insert(i.ident.clone(), t.ty.clone());
                    handler_argument_types.push(t.ty.clone());
                }
            }
            // syn::FnArg::Captured(cap) => match &cap.pat {
            //     syn::Pat::Ident(ident) => {
            //         handler_arguments.insert(ident.ident.clone(), cap.ty.clone());
            //         handler_argument_types.push(cap.ty.clone());
            //     }
            //     _ => unreachable!(),
            // },
            _ => unreachable!(),
        }
    });

    // Build the new handler
    let handler_argument_count = handler_arguments.len();
    let handler_function = if handler_arguments.is_empty() {
        match ast.sig.output {
            syn::ReturnType::Default => {
                if is_threaded {
                    quote! {
                        unsafe fn #handler(output: *mut arma_rs_libc::c_char, _: usize, _: Option<*mut *mut i8>, _: Option<usize>) {
                            std::thread::spawn(move || #name());
                        }
                    }
                } else {
                    quote! {
                        unsafe fn #handler(output: *mut arma_rs_libc::c_char, _: usize, _: Option<*mut *mut i8>, _: Option<usize>) {
                            #name();
                        }
                    }
                }
            }
            _ => {
                if is_threaded { panic!("Threaded functions can not return a value"); }

                quote! {
                    unsafe fn #handler(output: *mut arma_rs_libc::c_char, size: usize, _: Option<*mut *mut i8>, _: Option<usize>) {
                        write_str_to_ptr(#name().to_string(), output, size);
                    }
                }
            }
        }
    } else {
        match ast.sig.output {
            syn::ReturnType::Default => {
                if is_threaded {
                    quote! {
                        #[allow(clippy::transmute_ptr_to_ref)]
                        unsafe fn #handler(output: *mut arma_rs_libc::c_char, size: usize, args: Option<*mut *mut i8>, count: Option<usize>) {
                            // Build a C vec with the exact size requirement
                            let argv: &[*mut arma_rs_libc::c_char; #handler_argument_count] = std::mem::transmute(args.unwrap());
                            let mut argv: Vec<String> = argv.to_vec().into_iter().map(|s| {
                                std::ffi::CStr::from_ptr(s).to_str().unwrap().trim_matches('\"').to_owned()
                            }).collect();

                            argv.reverse();

                            std::thread::spawn(move || {
                                #name(
                                    #(
                                        #handler_argument_types::from_str(&argv.pop().unwrap()).unwrap()
                                    ),*
                                );
                            });
                        }
                    }
                } else {
                    quote! {
                        #[allow(clippy::transmute_ptr_to_ref)]
                        unsafe fn #handler(output: *mut arma_rs_libc::c_char, size: usize, args: Option<*mut *mut i8>, count: Option<usize>) {
                            // Build a C vec with the exact size requirement
                            let argv: &[*mut arma_rs_libc::c_char; #handler_argument_count] = std::mem::transmute(args.unwrap());
                            let mut argv: Vec<String> = argv.to_vec().into_iter().map(|s| {
                                std::ffi::CStr::from_ptr(s).to_str().unwrap().trim_matches('\"').to_owned()
                            }).collect();

                            argv.reverse();

                            #name(
                                #(
                                    #handler_argument_types::from_str(&argv.pop().unwrap()).unwrap()
                                ),*
                            );
                        }
                    }
                }
            }
            _ => {
                if is_threaded { panic!("Threaded functions can not return a value"); }

                quote! {
                    #[allow(clippy::transmute_ptr_to_ref)]
                    unsafe fn #handler(output: *mut arma_rs_libc::c_char, size: usize, args: Option<*mut *mut i8>, count: Option<usize>) {
                        // Build a C vec with the exact size requirement
                        let argv: &[*mut arma_rs_libc::c_char; #handler_argument_count] = std::mem::transmute(args.unwrap());
                        let mut argv: Vec<String> = argv.to_vec().into_iter().map(|s| {
                            std::ffi::CStr::from_ptr(s).to_str().unwrap().trim_matches('\"').to_owned()
                        }).collect();

                        argv.reverse();

                        let call_results = #name(
                            #(
                                #handler_argument_types::from_str(&argv.pop().unwrap()).unwrap()
                            ),*
                        );

                        log::debug!("R: {:?}", call_results.to_string());

                        write_str_to_ptr(call_results.to_string(), output, size);
                    }
                }
            }
        }
    };

    let expanded = quote! {
        #[allow(non_upper_case_globals)]
        static #info: FunctionInfo = FunctionInfo {
            handler: #handler,
            name: #name_string,
            thread: #is_threaded,
        };
        #handler_function
        #ast
    };

    if handler_arguments.is_empty() {
        PROXIES.lock().unwrap().push(name.to_string());
    } else {
        PROXIES_ARG.lock().unwrap().push(name.to_string());
    }

    TokenStream::from(expanded)
}

#[proc_macro_attribute]
/// Required for all extensions
///
/// Handles incoming information from Arma and calls the appropriate function.
/// Also can be used to run code at init.
///
/// ```ignore
/// use arma_rs::rv_handler;
///
/// #[rv_handler]
/// fn init() {
///     // Init code here
/// }
/// ```
pub fn rv_handler(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let ast = syn::parse_macro_input!(item as ItemFn);

    let proxies = (*PROXIES.lock().unwrap()).clone();
    let info: Vec<syn::Ident> = proxies
        .iter()
        .map(|s| syn::Ident::new(&format!("{}_info", s), proc_macro2::Span::call_site()))
        .collect();
    let proxies_arg = (*PROXIES_ARG.lock().unwrap()).clone();
    let infoarg: Vec<syn::Ident> = proxies_arg
        .iter()
        .map(|s| syn::Ident::new(&format!("{}_info", s), proc_macro2::Span::call_site()))
        .collect();

    let extern_type = if cfg!(windows) { "system" } else { "C" };

    let init = ast.sig.ident.clone();

    let expanded = quote! {
        use std::str::FromStr;
        use std::sync::Mutex;
        use arma_rs::libc as arma_rs_libc;

        #[derive(Debug)]
        pub struct FunctionInfo {
            name: &'static str,
            handler: unsafe fn(*mut arma_rs_libc::c_char, usize, Option<*mut *mut i8>, Option<usize>) -> (),
            thread: bool,
        }

        type ArmaCallback = extern #extern_type fn(*const arma_rs_libc::c_char, *const arma_rs_libc::c_char, *const arma_rs_libc::c_char) -> arma_rs_libc::c_int;

        static endpoint_proxies: &[&FunctionInfo] = &[#(&#info),*];
        static endpoint_proxies_arg: &[&FunctionInfo] = &[#(&#infoarg),*];
        static mut did_init: bool = false;
        static mut CALLBACK: Option<ArmaCallback> = None;

        #[allow(non_snake_case)]
        #[no_mangle]
        pub unsafe extern #extern_type fn RVExtensionVersion(
            output: *mut arma_rs_libc::c_char,
            size: arma_rs_libc::size_t
        ) {
            if !did_init { #init(); did_init = true; }

            // Tracing here because this is the first function called and trace! needs to be in a fn
            log::trace!("Proxies: {:?}", endpoint_proxies);
            log::trace!("ProxiesArgs: {:?}", endpoint_proxies_arg);

            write_str_to_ptr(env!("CARGO_PKG_VERSION").to_string(), output, size);
        }

        #[allow(non_snake_case)]
        #[no_mangle]
        pub unsafe extern #extern_type fn RVExtension(
            output: *mut arma_rs_libc::c_char,
            size: usize,
            function: *mut arma_rs_libc::c_char
        ) {
            if !did_init { #init(); did_init = true; }

            // Arma request with a function name. Find the definition and call it
            let rust_function_name = std::ffi::CStr::from_ptr(function).to_str().unwrap();
            for info in endpoint_proxies.iter() {
                if info.name != rust_function_name { continue; }

                // We've found the handler, call it
                (info.handler)(output, size, None, None);
                return;
            }

            log::error!("[rv_handler] Failed to find endpoint \"{}\"", rust_function_name);
        }

        #[allow(non_snake_case)]
        #[no_mangle]
        pub unsafe extern #extern_type fn RVExtensionArgs(
            output: *mut arma_rs_libc::c_char,
            size: usize,
            function: *mut arma_rs_libc::c_char,
            args: *mut *mut arma_rs_libc::c_char,
            arg_count: usize
        ) {
            if !did_init { #init(); did_init = true; }

            let rust_function_name = std::ffi::CStr::from_ptr(function).to_str().unwrap();
            for info in endpoint_proxies_arg.iter() {
                if info.name != rust_function_name { continue; }

                // We found the function. Call it with the arguments
                (info.handler)(output, size, Some(args), Some(arg_count));
                return;
            }

            log::error!("[rv_handler] Failed to find endpoint \"{}\"", rust_function_name);
        }

        #[allow(non_snake_case)]
        #[no_mangle]
        pub unsafe extern #extern_type fn RVExtensionRegisterCallback(
            callback: ArmaCallback
        ) {
            CALLBACK = Some(callback);
        }

        unsafe fn rv_send_callback(
            name: *const arma_rs_libc::c_char,
            function: *const arma_rs_libc::c_char,
            data: *const arma_rs_libc::c_char
        ) {
            if let Some(callback) = CALLBACK {
                loop {
                    if callback(name, function, data) >= 0 { break; }

                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        }

        /// Copies an ASCII rust string into a memory buffer as a C string.
        /// Performs necessary validation, including:
        /// * Ensuring the string is ASCII
        /// * Ensuring the string has no null bytes except at the end
        /// * Making sure string length doesn't exceed the buffer.
        /// # Returns
        /// :Option with the number of ASCII characters written - *excludes the C null terminator*
        /// Extracted from: https://github.com/Spoffy/Rust-Arma-Extension-Example/blob/5fc61340a1572ddecd9f8caf5458fd4faaf28e20/src/lib.rs#L95
        unsafe fn write_str_to_ptr(
            string: String,
            ptr: *mut arma_rs_libc::c_char,
            buf_size: arma_rs_libc::size_t
        ) -> Option<usize> {
            // We shouldn't encode non-ascii string as C strings, things will get weird. Better to abort, I think.
            if !string.is_ascii() { return None };

            // This should never fail, honestly - we'd have to have manually added null bytes or something.
            let cstr = std::ffi::CString::new(string).ok()?;
            let cstr_bytes = cstr.as_bytes();

            // C Strings end in null bytes. We want to make sure we always write a valid string.
            // So we want to be able to always write a null byte at the end.
            let amount_to_copy = std::cmp::min(cstr_bytes.len(), buf_size - 1);

            // We provide a guarantee to our unsafe code, that we'll never pass anything too large.
            // In reality, I can't see this ever happening.
            if amount_to_copy > isize::MAX as usize { return None }

            log::trace!("[rv_handler] Writing {:?} with size {}", cstr, amount_to_copy);

            // We'll never copy the whole string here - it will always be missing the null byte.
            ptr.copy_from(cstr.as_ptr(), amount_to_copy);

            // Add our null byte at the end
            ptr.add(amount_to_copy).write(0x00);
            Some(amount_to_copy)
        }

        #ast
    };

    TokenStream::from(expanded)
}
