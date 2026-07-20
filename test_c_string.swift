import Foundation
func test(_ ptr: UnsafePointer<CChar>!) {
    print(ptr == nil ? "nil" : String(cString: ptr))
}
let s = "Hello"
test(s)
let s2: String? = nil
test(s2)
